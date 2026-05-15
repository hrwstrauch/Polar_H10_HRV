#!/usr/bin/env python3
"""
polar_h10_hrv.py — Real-time HRV monitoring with the Polar H10 chest strap.

Connects to a Polar H10 over Bluetooth Low Energy (BLE), subscribes to the
standard Heart Rate Service, parses heart rate and RR intervals, computes
rolling HRV metrics (RMSSD, SDNN), and optionally shows a live plot.
Every beat is logged to CSV for offline spectral analysis.

The Heart Rate Measurement characteristic (UUID 0x2A37) is parsed per the
Bluetooth GATT specification:

    Byte 0       flags
                   bit 0  : 0 = HR is uint8, 1 = HR is uint16
                   bit 1-2: sensor contact status
                   bit 3  : energy expended field present
                   bit 4  : RR intervals present
    Byte 1[-2]   heart rate value (8- or 16-bit unsigned, little-endian)
    [optional]   energy expended (uint16 LE, kJ)        — if bit 3 set
    [optional]   one or more RR intervals (uint16 LE,
                 units of 1/1024 s)                     — if bit 4 set

RR intervals come in units of 1/1024 second; convert to milliseconds by
dividing by 1.024. The Polar H10 commonly reports more than one RR interval
per notification when the previous beat was missed in a notification cycle,
so the parser must consume them all.

Dependencies
------------
    pip install bleak numpy matplotlib

Usage
-----
    python polar_h10_hrv.py                # scan, connect, live plot, log CSV
    python polar_h10_hrv.py --no-plot      # console + CSV only
    python polar_h10_hrv.py --scan         # list nearby BLE devices and exit
    python polar_h10_hrv.py --address AA:BB:CC:DD:EE:FF
                                           # skip scan, connect by address
    python polar_h10_hrv.py --outdir ~/hrv_logs

Stop the program with Ctrl-C; the CSV is flushed on every beat so partial
sessions are recoverable.

Author note
-----------
The capture loop runs in a background thread using its own asyncio event
loop, while matplotlib animation runs on the main thread. The Session
object is shared via a threading.Lock — needed because deque.append and
the metric computation can race in principle, even though for ~1 Hz beat
data on CPython the GIL would make this unlikely to matter in practice.
"""

import argparse
import asyncio
import csv
import struct
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
from bleak import BleakClient, BleakScanner

# --------------------------------------------------------------------------
# BLE constants
# --------------------------------------------------------------------------
HEART_RATE_SERVICE_UUID     = "0000180d-0000-1000-8000-00805f9b34fb"
HEART_RATE_MEASUREMENT_UUID = "00002a37-0000-1000-8000-00805f9b34fb"
DEVICE_NAME_PREFIX          = "Polar H10"


# --------------------------------------------------------------------------
# HR measurement parser
# --------------------------------------------------------------------------
def parse_hr_measurement(data: bytearray):
    """
    Parse a Heart Rate Measurement notification.
    Returns (heart_rate_bpm, [rr_ms, ...]).
    """
    flags = data[0]
    hr_16bit       = bool(flags & 0x01)
    energy_present = bool(flags & 0x08)
    rr_present     = bool(flags & 0x10)

    idx = 1
    if hr_16bit:
        hr = struct.unpack_from("<H", data, idx)[0]
        idx += 2
    else:
        hr = data[idx]
        idx += 1

    if energy_present:
        idx += 2  # skip energy expended

    rr_ms = []
    if rr_present:
        while idx + 1 < len(data):
            rr_units = struct.unpack_from("<H", data, idx)[0]
            rr_ms.append(rr_units / 1.024)  # 1/1024 s units -> milliseconds
            idx += 2

    return hr, rr_ms


# --------------------------------------------------------------------------
# HRV metrics
# --------------------------------------------------------------------------
def rmssd(rr_series):
    """Root mean square of successive differences (vagal-tone proxy)."""
    if len(rr_series) < 2:
        return float("nan")
    diffs = np.diff(rr_series)
    return float(np.sqrt(np.mean(diffs ** 2)))


def sdnn(rr_series):
    """Standard deviation of NN intervals (overall HRV)."""
    if len(rr_series) < 2:
        return float("nan")
    return float(np.std(rr_series, ddof=1))


def pnn50(rr_series):
    """Percentage of consecutive RR intervals differing by more than 50 ms."""
    if len(rr_series) < 2:
        return float("nan")
    diffs = np.abs(np.diff(rr_series))
    return float(np.mean(diffs > 50.0) * 100.0)


# --------------------------------------------------------------------------
# Session state (shared between BLE thread and plot thread)
# --------------------------------------------------------------------------
class Session:
    def __init__(self, csv_path: Path, max_beats: int = 1800,
                 comp_window: float = 60.0):
        # max_beats=1800 covers ~30 minutes at 60 bpm — adjust to taste.
        self.start = time.time()
        self.comp_window = comp_window
        self.lock = threading.Lock()
        self.rr_buffer = deque(maxlen=max_beats)
        self.hr_buffer = deque(maxlen=max_beats)
        self.t_buffer  = deque(maxlen=max_beats)
        self.beat_count = 0
        self.rmssd_hist    = deque(maxlen=max_beats)
        self.sdnn_hist     = deque(maxlen=max_beats)
        self.pnn50_hist    = deque(maxlen=max_beats)
        self.metric_t_hist = deque(maxlen=max_beats)

        self.csv_path = csv_path
        self.csv_file = open(csv_path, "w", newline="")
        self.csv = csv.writer(self.csv_file)
        self.csv.writerow(["t_session_s", "hr_bpm", "rr_ms"])

    def _windowed_rr(self):
        """Return RR intervals (ms) that fall within the comp_window. Call with lock held."""
        cutoff = time.time() - self.start - self.comp_window
        return [r for r, t in zip(self.rr_buffer, self.t_buffer) if t >= cutoff]

    def add(self, hr, rr_list):
        with self.lock:
            for rr in rr_list:
                t = time.time() - self.start
                self.rr_buffer.append(rr)
                self.hr_buffer.append(hr)
                self.t_buffer.append(t)
                self.beat_count += 1
                self.csv.writerow([f"{t:.3f}", hr, f"{rr:.1f}"])
            self.csv_file.flush()
            if rr_list:
                win = self._windowed_rr()
                self.rmssd_hist.append(rmssd(win))
                self.sdnn_hist.append(sdnn(win))
                self.pnn50_hist.append(pnn50(win))
                self.metric_t_hist.append(time.time() - self.start)

    def snapshot(self):
        """Return a thread-safe copy of the buffers and current metrics."""
        with self.lock:
            t  = list(self.t_buffer)
            hr = list(self.hr_buffer)
            rr = list(self.rr_buffer)
            n  = self.beat_count
            mt = list(self.metric_t_hist)
            rm = list(self.rmssd_hist)
            sd = list(self.sdnn_hist)
            pn = list(self.pnn50_hist)
        cutoff = time.time() - self.start - self.comp_window
        win = [r for r, ti in zip(rr, t) if ti >= cutoff]
        return {
            "t":  t,
            "hr": hr,
            "rr": rr,
            "n_beats":    n,
            "elapsed_s":  time.time() - self.start,
            "rmssd_ms":   rmssd(win),
            "sdnn_ms":    sdnn(win),
            "pnn50_pct":  pnn50(win),
            "metric_t":   mt,
            "rmssd_hist": rm,
            "sdnn_hist":  sd,
            "pnn50_hist": pn,
        }

    def close(self):
        self.csv_file.close()


# --------------------------------------------------------------------------
# BLE scan and stream
# --------------------------------------------------------------------------
async def find_polar_h10(timeout: float = 10.0):
    print(f"Scanning for Polar H10 ({timeout:.0f} s)...", flush=True)
    devices = await BleakScanner.discover(timeout=timeout)
    for d in devices:
        if d.name and d.name.startswith(DEVICE_NAME_PREFIX):
            print(f"Found {d.name} at {d.address}", flush=True)
            return d
    print("No Polar H10 found. Make sure it is worn (skin contact wakes "
          "the strap), Bluetooth is on, and you are within ~10 m.",
          flush=True)
    return None


async def stream(address: str, session: Session, verbose: bool = True):
    async with BleakClient(address) as client:
        print(f"Connected to {address}", flush=True)

        def handler(_sender, data):
            try:
                hr, rr_list = parse_hr_measurement(bytearray(data))
            except Exception as e:
                print(f"\nParse error: {e!r}  raw={bytes(data).hex()}",
                      flush=True)
                return
            session.add(hr, rr_list)
            if verbose:
                snap = session.snapshot()
                print(
                    f"\r[t={snap['elapsed_s']:7.1f}s  "
                    f"beats={snap['n_beats']:5d}] "
                    f"HR={hr:3d}  RR={['%.0f' % x for x in rr_list]}  "
                    f"RMSSD={snap['rmssd_ms']:6.1f}  "
                    f"SDNN={snap['sdnn_ms']:6.1f}  "
                    f"pNN50={snap['pnn50_pct']:5.1f}%      ",
                    end="", flush=True,
                )

        await client.start_notify(HEART_RATE_MEASUREMENT_UUID, handler)
        try:
            while True:
                await asyncio.sleep(1.0)
        finally:
            try:
                await client.stop_notify(HEART_RATE_MEASUREMENT_UUID)
            except Exception:
                pass


# --------------------------------------------------------------------------
# Live plot (matplotlib animation on main thread)
# --------------------------------------------------------------------------
def run_plot(session: Session):
    import matplotlib.pyplot as plt
    import matplotlib.animation as anim

    # --- Figure 1: HR and RR time series ---
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5),
                                    gridspec_kw={"height_ratios": [1, 1]})
    fig1.suptitle("Polar H10 — live HRV monitor", fontsize=13)

    line_hr, = ax1.plot([], [], "-", color="#1F3864", linewidth=1.2)
    ax1.set_xlabel("session time [s]")
    ax1.set_ylabel("instantaneous HR [bpm]")
    ax1.grid(True, alpha=0.3)

    # RR intervals (Poincare-style scatter would be nicer; use
    # time series for now — interpretable at a glance.)
    line_rr, = ax2.plot([], [], ".", color="#2E5395", markersize=3)
    ax2.set_xlabel("beat index")
    ax2.set_ylabel("RR interval [ms]")
    ax2.grid(True, alpha=0.3)

    txt = ax1.text(
        0.012, 0.96, "", transform=ax1.transAxes,
        fontsize=10, family="monospace", verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#888", alpha=0.85),
    )

    # --- Figure 2: rolling RMSSD and SDNN (bio-feedback window) ---
    fig2, (ax3, ax4, ax5) = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True,
                                          gridspec_kw={"height_ratios": [1, 1, 1]})
    fig2.suptitle(
        f"HRV metrics  —  sliding window {session.comp_window:.0f} s",
        fontsize=13,
    )

    line_rmssd, = ax3.plot([], [], "-", color="#C00000", linewidth=1.4)
    ax3.set_ylabel("RMSSD [ms]")
    ax3.grid(True, alpha=0.3)

    line_sdnn, = ax4.plot([], [], "-", color="#375623", linewidth=1.4)
    ax4.set_ylabel("SDNN [ms]")
    ax4.grid(True, alpha=0.3)

    line_pnn50, = ax5.plot([], [], "-", color="#7030A0", linewidth=1.4)
    ax5.set_ylabel("pNN50 [%]")
    ax5.set_xlabel("session time [s]")
    ax5.grid(True, alpha=0.3)

    fig1.tight_layout()
    fig2.tight_layout()

    def update(_frame):
        snap = session.snapshot()
        if snap["n_beats"] == 0:
            return line_hr, line_rr, txt, line_rmssd, line_sdnn, line_pnn50

        # fig1 updates
        line_hr.set_data(snap["t"], snap["hr"])
        line_rr.set_data(range(len(snap["rr"])), snap["rr"])
        ax1.relim(); ax1.autoscale_view()
        ax2.relim(); ax2.autoscale_view()
        txt.set_text(
            f"beats : {snap['n_beats']:5d}\n"
            f"elapsed: {snap['elapsed_s']:6.1f} s\n"
            f"RMSSD : {snap['rmssd_ms']:6.1f} ms\n"
            f"SDNN  : {snap['sdnn_ms']:6.1f} ms\n"
            f"pNN50 : {snap['pnn50_pct']:5.1f} %"
        )

        # fig2 updates
        mt = snap["metric_t"]
        if mt:
            line_rmssd.set_data(mt, snap["rmssd_hist"])
            line_sdnn.set_data(mt, snap["sdnn_hist"])
            line_pnn50.set_data(mt, snap["pnn50_hist"])
            ax3.relim(); ax3.autoscale_view()
            ax4.relim(); ax4.autoscale_view()
            ax5.relim(); ax5.autoscale_view()
            fig2.canvas.draw_idle()

        return line_hr, line_rr, txt, line_rmssd, line_sdnn, line_pnn50

    ani = anim.FuncAnimation(
        fig1, update, interval=500, blit=False, cache_frame_data=False,
    )
    plt.show()
    return ani  # keep reference alive


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
async def amain(args):
    if args.scan:
        print("Scanning BLE devices for 10 s ...")
        devices = await BleakScanner.discover(timeout=10.0)
        if not devices:
            print("  (no devices found)")
            return
        for d in devices:
            print(f"  {d.address}   {d.name or '<unnamed>'}")
        return

    # Resolve device
    if args.address:
        address = args.address
    else:
        device = await find_polar_h10()
        if device is None:
            sys.exit(1)
        address = device.address

    # Set up CSV log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"polar_h10_{timestamp}.csv"
    session = Session(csv_path, comp_window=args.comp_window)
    print(f"Logging to {csv_path}")

    if args.no_plot:
        # Single-thread asyncio version
        try:
            await stream(address, session)
        except KeyboardInterrupt:
            pass
        finally:
            session.close()
            snap = session.snapshot()
            print()
            print(f"Saved {snap['n_beats']} beats over "
                  f"{snap['elapsed_s']:.1f} s to {csv_path}")
    else:
        # Run BLE in a background thread; matplotlib on main thread.
        stop_event = threading.Event()

        def ble_worker():
            try:
                asyncio.run(stream(address, session, verbose=False))
            except Exception as e:
                print(f"\nBLE thread error: {e!r}", flush=True)
            finally:
                stop_event.set()

        t = threading.Thread(target=ble_worker, daemon=True)
        t.start()
        try:
            run_plot(session)
        finally:
            session.close()
            snap = session.snapshot()
            print(f"\nSaved {snap['n_beats']} beats over "
                  f"{snap['elapsed_s']:.1f} s to {csv_path}")


def main():
    p = argparse.ArgumentParser(
        description="Polar H10 -> live HRV streamer (HR + RR over BLE)."
    )
    p.add_argument("--address", help="BLE address of the strap (skip scan)")
    p.add_argument("--scan", action="store_true",
                   help="List nearby BLE devices and exit")
    p.add_argument("--no-plot", action="store_true",
                   help="Console + CSV only, no matplotlib window")
    p.add_argument("--outdir", default=".",
                   help="Directory for CSV log file (default: current dir)")
    p.add_argument("--comp-window", type=float, default=60.0, metavar="SEC",
                   help="Sliding window length in seconds for RMSSD/SDNN "
                        "(default: 60)")
    args = p.parse_args()
    try:
        asyncio.run(amain(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
