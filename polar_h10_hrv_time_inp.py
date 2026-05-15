#!/usr/bin/env python3
"""
polar_h10_hrv.py — Real-time HRV monitoring with the Polar H10 chest strap.

Connects to a Polar H10 over Bluetooth Low Energy (BLE), subscribes to the
standard Heart Rate Service, parses heart rate and RR intervals, computes
rolling HRV metrics (RMSSD, SDNN), and optionally shows a live plot.
Every beat is logged to CSV for offline spectral analysis.

Two timing parameters control the session:

    --delay T_delay   Discard beats for the first T_delay seconds after
                      connection.  Use this to let the sensor settle and
                      the user to reach a steady state before recording
                      begins.  Default: 0 (log from the first beat).

    --run   T_run     Terminate automatically T_run seconds after the CSV
                      recording starts (i.e. after the delay has elapsed).
                      Default: 0 (run until Ctrl-C).

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
dividing by 1.024.

Dependencies
------------
    pip install bleak numpy matplotlib

Usage
-----
    python polar_h10_hrv.py                        # scan, connect, live plot, log CSV
    python polar_h10_hrv.py --no-plot              # console + CSV only
    python polar_h10_hrv.py --scan                 # list nearby BLE devices and exit
    python polar_h10_hrv.py --address AA:BB:CC:DD:EE:FF
    python polar_h10_hrv.py --delay 60 --run 300   # 60 s settle, then record 300 s
    python polar_h10_hrv.py --outdir ~/hrv_logs

Stop the program with Ctrl-C; the CSV is flushed on every beat so partial
sessions are recoverable.
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
    def __init__(self, csv_path: Path, t_delay: float = 0.0,
                 t_run: float = 0.0, max_beats: int = 1800):
        """
        t_delay : seconds after connection to discard beats before CSV logging.
        t_run   : seconds of CSV recording after which the session stops
                  autonomously.  0 means run until Ctrl-C.
        """
        self.connect_time = time.time()
        self.t_delay = t_delay
        self.t_run   = t_run

        # Set once the delay has elapsed; marks the recording start.
        self._record_start: float | None = None
        # Set to signal the BLE loop to stop.
        self.done = threading.Event()

        self.lock = threading.Lock()
        self.rr_buffer = deque(maxlen=max_beats)
        self.hr_buffer = deque(maxlen=max_beats)
        self.t_buffer  = deque(maxlen=max_beats)
        self.beat_count = 0

        self.csv_path = csv_path
        self.csv_file = open(csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["t_record_s", "hr_bpm", "rr_ms"])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _elapsed_since_connect(self) -> float:
        return time.time() - self.connect_time

    def _recording(self) -> bool:
        """True once the delay has elapsed."""
        return self._record_start is not None

    def _t_record(self) -> float:
        """Seconds since recording started (undefined before recording)."""
        return time.time() - self._record_start

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def add(self, hr, rr_list):
        """
        Called from the BLE notification handler for every beat.
        Beats arriving during the delay window are counted but not stored.
        """
        now = time.time()

        with self.lock:
            elapsed = now - self.connect_time

            # --- Still in delay window ---
            if elapsed < self.t_delay:
                return  # discard silently

            # --- Transition: delay just elapsed ---
            if self._record_start is None:
                self._record_start = now
                print(f"\n[t={elapsed:.1f}s] Delay elapsed — CSV recording started.",
                      flush=True)

            # --- Recording ---
            t_rec = now - self._record_start
            for rr in rr_list:
                self.rr_buffer.append(rr)
                self.hr_buffer.append(hr)
                self.t_buffer.append(t_rec)
                self.beat_count += 1
                self.csv_writer.writerow([f"{t_rec:.3f}", hr, f"{rr:.1f}"])
            self.csv_file.flush()

            # --- Check run-time limit ---
            if self.t_run > 0 and t_rec >= self.t_run:
                self.done.set()

    def snapshot(self):
        """Return a thread-safe copy of the buffers and current metrics."""
        with self.lock:
            t  = list(self.t_buffer)
            hr = list(self.hr_buffer)
            rr = list(self.rr_buffer)
            n  = self.beat_count
            elapsed_connect = self._elapsed_since_connect()
            in_delay = not self._recording()
            t_rec = self._t_record() if self._recording() else 0.0
            remaining = (
                max(0.0, self.t_run - t_rec) if self.t_run > 0 else None
            )
        return {
            "t":  t,
            "hr": hr,
            "rr": rr,
            "n_beats":        n,
            "elapsed_connect": elapsed_connect,
            "t_record_s":     t_rec,
            "in_delay":       in_delay,
            "remaining_s":    remaining,
            "rmssd_ms":  rmssd(rr),
            "sdnn_ms":   sdnn(rr),
            "pnn50_pct": pnn50(rr),
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
        if session.t_delay > 0:
            print(f"Delay active: discarding beats for {session.t_delay:.1f} s "
                  f"before CSV recording begins.", flush=True)
        if session.t_run > 0:
            print(f"Run limit: recording will stop after {session.t_run:.1f} s.",
                  flush=True)

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
                if snap["in_delay"]:
                    delay_left = max(0.0, session.t_delay - snap["elapsed_connect"])
                    print(
                        f"\r[DELAY  {delay_left:6.1f}s remaining] "
                        f"HR={hr:3d}  RR={['%.0f' % x for x in rr_list]}    ",
                        end="", flush=True,
                    )
                else:
                    run_info = (
                        f"  rem={snap['remaining_s']:6.1f}s"
                        if snap["remaining_s"] is not None else ""
                    )
                    print(
                        f"\r[t={snap['t_record_s']:7.1f}s  "
                        f"beats={snap['n_beats']:5d}{run_info}] "
                        f"HR={hr:3d}  RR={['%.0f' % x for x in rr_list]}  "
                        f"RMSSD={snap['rmssd_ms']:6.1f}  "
                        f"SDNN={snap['sdnn_ms']:6.1f}  "
                        f"pNN50={snap['pnn50_pct']:5.1f}%      ",
                        end="", flush=True,
                    )

        await client.start_notify(HEART_RATE_MEASUREMENT_UUID, handler)
        try:
            # Poll done-event at 1 Hz instead of sleeping indefinitely.
            while not session.done.is_set():
                await asyncio.sleep(1.0)
            print("\nRun-time limit reached — stopping.", flush=True)
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

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5),
                                   gridspec_kw={"height_ratios": [1, 1]})
    fig.suptitle("Polar H10 — live HRV monitor", fontsize=13)

    line_hr, = ax1.plot([], [], "-", color="#1F3864", linewidth=1.2)
    ax1.set_xlabel("recording time [s]")
    ax1.set_ylabel("instantaneous HR [bpm]")
    ax1.grid(True, alpha=0.3)

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

    def update(_frame):
        snap = session.snapshot()

        # Close the plot automatically when the run limit is reached.
        if session.done.is_set():
            plt.close(fig)
            return line_hr, line_rr, txt

        if snap["in_delay"]:
            delay_left = max(0.0, session.t_delay - snap["elapsed_connect"])
            txt.set_text(f"DELAY  {delay_left:.0f} s remaining\n"
                         f"(waiting before recording)")
            return line_hr, line_rr, txt

        if snap["n_beats"] == 0:
            return line_hr, line_rr, txt

        line_hr.set_data(snap["t"], snap["hr"])
        line_rr.set_data(range(len(snap["rr"])), snap["rr"])
        ax1.relim(); ax1.autoscale_view()
        ax2.relim(); ax2.autoscale_view()

        rem_line = (
            f"remain : {snap['remaining_s']:6.1f} s\n"
            if snap["remaining_s"] is not None else ""
        )
        txt.set_text(
            f"beats  : {snap['n_beats']:5d}\n"
            f"rec    : {snap['t_record_s']:6.1f} s\n"
            f"{rem_line}"
            f"RMSSD  : {snap['rmssd_ms']:6.1f} ms\n"
            f"SDNN   : {snap['sdnn_ms']:6.1f} ms\n"
            f"pNN50  : {snap['pnn50_pct']:5.1f} %"
        )
        return line_hr, line_rr, txt

    ani = anim.FuncAnimation(
        fig, update, interval=500, blit=False, cache_frame_data=False,
    )
    plt.tight_layout()
    plt.show()
    return ani


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
    session = Session(csv_path, t_delay=args.delay, t_run=args.run)
    print(f"Logging to {csv_path}")

    if args.no_plot:
        try:
            await stream(address, session)
        except KeyboardInterrupt:
            pass
        finally:
            session.close()
            snap = session.snapshot()
            print()
            print(f"Saved {snap['n_beats']} beats over "
                  f"{snap['t_record_s']:.1f} s (recording) / "
                  f"{snap['elapsed_connect']:.1f} s (total) to {csv_path}")
    else:
        stop_event = threading.Event()

        def ble_worker():
            try:
                asyncio.run(stream(address, session, verbose=False))
            except Exception as e:
                print(f"\nBLE thread error: {e!r}", flush=True)
            finally:
                stop_event.set()
                session.done.set()   # ensure plot closes too

        t = threading.Thread(target=ble_worker, daemon=True)
        t.start()
        try:
            run_plot(session)
        finally:
            session.done.set()   # in case user closed the plot window
            session.close()
            snap = session.snapshot()
            print(f"\nSaved {snap['n_beats']} beats over "
                  f"{snap['t_record_s']:.1f} s (recording) / "
                  f"{snap['elapsed_connect']:.1f} s (total) to {csv_path}")


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
    p.add_argument("--delay", type=float, default=0.0, metavar="T_delay",
                   help="Seconds to discard beats after connection before "
                        "CSV recording starts (default: 0)")
    p.add_argument("--run", type=float, default=0.0, metavar="T_run",
                   help="Seconds of recording after which the program exits "
                        "autonomously (default: 0 = run until Ctrl-C)")
    args = p.parse_args()
    try:
        asyncio.run(amain(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
