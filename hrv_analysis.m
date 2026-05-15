function results = hrv_analysis(csvfile, opts)
% HRV_ANALYSIS  HRV analysis from a Polar H10 RR-interval CSV.
%
%   results = HRV_ANALYSIS(csvfile)
%   results = HRV_ANALYSIS(csvfile, opts)
%
%   CSV is assumed to contain columns named hr_bpm and rr_ms (other columns
%   are ignored). Polar H10 reports one rr_ms value per detected R-R event.
%
%   Outputs (struct):
%     .timeDomain      mean RR, SDNN, RMSSD, pNN50
%     .freqDomain      VLF, LF, HF absolute power [ms^2], LF/HF, normalised units
%     .poincare        SD1, SD2, SD1/SD2 (ellipse fit on first-return map)
%     .dfa             alpha1 (short-range scaling, default 4..16 beats)
%     .figures         handle vector of generated figures
%
%   Default options (override by passing opts struct):
%     opts.fs          = 4;          % resampling rate for Welch (Hz)
%     opts.windowSec   = 256;        % Welch segment length (s) — needs >= ~5 min
%     opts.overlap     = 0.5;        % Welch overlap fraction
%     opts.rrMin       = 300;        % physiological RR floor (ms)
%     opts.rrMax       = 2000;       % physiological RR ceiling (ms)
%     opts.rejectFrac  = 0.20;       % reject beats with |dRR|/RR > rejectFrac
%     opts.detrend     = 'linear';   % 'linear' | 'mean' | 'none'
%     opts.dfaScales   = 4:1:16;     % scales (beats) for alpha1
%     opts.plot        = true;
%
%   Reference bands (Task Force 1996):
%     VLF : 0.0033 – 0.04  Hz
%     LF  : 0.04   – 0.15  Hz
%     HF  : 0.15   – 0.40  Hz
%
%   Notes from a control-theory perspective:
%     - RR series is unevenly sampled (sample = one beat). Welch needs a
%       uniform grid, so we resample via shape-preserving cubic interpolation
%       onto t = 0:1/fs:T. fs=4 Hz captures the HF band well below Nyquist.
%     - Linear detrending is the standard pre-Welch step; it removes the
%       VLF "DC ramp" caused by slow circadian/baroreflex drift that
%       otherwise smears across the spectrum.
%     - SD1 and SD2 are the minor/major axes of the 1-lag Poincaré ellipse.
%       SD1 = sqrt(0.5 * Var(dRR))  (= RMSSD/sqrt(2))  -> short-term, vagal
%       SD2 = sqrt(2*Var(RR) - 0.5*Var(dRR))           -> long-term, mixed
%     - DFA alpha1 is the slope of log F(n) vs log n for n in opts.dfaScales,
%       where F(n) is the rms fluctuation of the locally-detrended integrated
%       series. alpha1 ~ 1.0 indicates 1/f ("pink") healthy organisation.

    if nargin < 2, opts = struct(); end
    opts = setDefaults(opts);

    %% ---------- 1. Load CSV ----------
    T = readtable(csvfile);
    vn = lower(string(T.Properties.VariableNames));
    rrCol = find(contains(vn, "rr"),   1, 'first');
    assert(~isempty(rrCol), "Could not find an rr_ms column in %s", csvfile);
    rr_ms = T{:, rrCol};
    rr_ms = rr_ms(~isnan(rr_ms));

    %% ---------- 2. Artifact rejection (NN series) ----------
    % Step 1: physiological window
    keep = (rr_ms >= opts.rrMin) & (rr_ms <= opts.rrMax);
    rr   = rr_ms(keep);

    % Step 2: successive-difference filter — reject any beat that differs
    %         from its neighbour by more than rejectFrac (default 20%).
    nn = rr;
    drr = abs(diff(nn));
    badIdx = false(size(nn));
    badIdx(2:end) = drr ./ nn(1:end-1) > opts.rejectFrac;
    nn(badIdx) = [];

    nRej = numel(rr_ms) - numel(nn);
    fprintf("Loaded %d RR samples; rejected %d (%.1f%%); kept %d NN.\n", ...
            numel(rr_ms), nRej, 100*nRej/numel(rr_ms), numel(nn));

    % Build beat-time axis (cumulative)
    t_nn = cumsum(nn) / 1000;          % seconds from session start
    t_nn = t_nn - t_nn(1);

    %% ---------- 3. Time-domain ----------
    td.meanRR_ms = mean(nn);
    td.meanHR_bpm = 60000 / td.meanRR_ms;
    td.SDNN_ms   = std(nn);
    td.RMSSD_ms  = sqrt(mean(diff(nn).^2));
    td.pNN50_pct = 100 * sum(abs(diff(nn)) > 50) / (numel(nn)-1);

    %% ---------- 4. Frequency-domain (Welch PSD) ----------
    % Resample onto uniform grid
    fs  = opts.fs;
    tu  = (0:1/fs:t_nn(end))';
    rru = interp1(t_nn, nn, tu, 'pchip');     % ms

    % Detrend before Welch
    switch lower(opts.detrend)
        case 'linear', rrd = detrend(rru, 1);
        case 'mean',   rrd = rru - mean(rru);
        otherwise,     rrd = rru;
    end

    % Welch
    winLen = min(round(opts.windowSec*fs), numel(rrd));
    nover  = round(opts.overlap * winLen);
    nfft   = max(2^nextpow2(winLen), 1024);
    [Pxx, f] = pwelch(rrd, hann(winLen), nover, nfft, fs);
    % Pxx is in (ms^2)/Hz so band integrals are in ms^2.

    bandPow = @(lo,hi) trapz(f(f>=lo & f<=hi), Pxx(f>=lo & f<=hi));
    fd.VLF_ms2 = bandPow(0.0033, 0.04);
    fd.LF_ms2  = bandPow(0.04,   0.15);
    fd.HF_ms2  = bandPow(0.15,   0.40);
    fd.TP_ms2  = fd.VLF_ms2 + fd.LF_ms2 + fd.HF_ms2;
    fd.LF_HF   = fd.LF_ms2 / fd.HF_ms2;
    lfhfTot    = fd.LF_ms2 + fd.HF_ms2;
    fd.LFnu    = 100 * fd.LF_ms2 / lfhfTot;
    fd.HFnu    = 100 * fd.HF_ms2 / lfhfTot;

    %% ---------- 5. Poincaré SD1 / SD2 ----------
    x = nn(1:end-1);
    y = nn(2:end);
    dxy = x - y;          % rotated coordinate aligned with minor axis
    sxy = x + y;          % rotated coordinate aligned with major axis
    pc.SD1 = sqrt(0.5) * std(dxy);
    pc.SD2 = sqrt(2*var(nn) - 0.5*var(diff(nn)));
    pc.SD1_SD2 = pc.SD1 / pc.SD2;
    pc.ellipseArea = pi * pc.SD1 * pc.SD2;   % ms^2

    %% ---------- 6. DFA alpha1 ----------
    [dfa.alpha1, dfa.n, dfa.F] = computeDFA(nn, opts.dfaScales);

    %% ---------- 7. Report ----------
    fprintf("\n--- Time domain ---\n");
    fprintf("  meanHR  = %6.1f bpm   meanRR = %6.1f ms\n", td.meanHR_bpm, td.meanRR_ms);
    fprintf("  SDNN    = %6.1f ms    RMSSD  = %6.1f ms    pNN50 = %5.2f %%\n", ...
            td.SDNN_ms, td.RMSSD_ms, td.pNN50_pct);
    fprintf("--- Frequency domain (Welch) ---\n");
    fprintf("  VLF     = %8.1f ms^2 (%4.1f%%)\n", fd.VLF_ms2, 100*fd.VLF_ms2/fd.TP_ms2);
    fprintf("  LF      = %8.1f ms^2 (%4.1f%%)\n", fd.LF_ms2,  100*fd.LF_ms2 /fd.TP_ms2);
    fprintf("  HF      = %8.1f ms^2 (%4.1f%%)\n", fd.HF_ms2,  100*fd.HF_ms2 /fd.TP_ms2);
    fprintf("  LF/HF   = %6.2f    LFnu = %5.1f    HFnu = %5.1f\n", fd.LF_HF, fd.LFnu, fd.HFnu);
    fprintf("--- Poincaré ---\n");
    fprintf("  SD1     = %6.1f ms    SD2 = %6.1f ms    SD1/SD2 = %5.3f\n", ...
            pc.SD1, pc.SD2, pc.SD1_SD2);
    fprintf("--- DFA ---\n");
    fprintf("  alpha1  = %6.3f over n = [%d .. %d] beats\n", ...
            dfa.alpha1, min(opts.dfaScales), max(opts.dfaScales));

    %% ---------- 8. Plots ----------
    figs = [];
    if opts.plot
        figs(end+1) = figure('Name','RR / NN tachogram','Color','w');
        plot(t_nn/60, nn, '-', 'LineWidth', 0.6); grid on;
        xlabel('Time [min]'); ylabel('NN interval [ms]');
        title('NN tachogram (after artifact rejection)');

        figs(end+1) = figure('Name','Welch PSD','Color','w');
        semilogy(f, Pxx, 'LineWidth', 1.2); grid on; hold on;
        shadeBand(0.0033, 0.04, [0.85 0.85 1.00], 'VLF');
        shadeBand(0.04,   0.15, [0.80 1.00 0.80], 'LF');
        shadeBand(0.15,   0.40, [1.00 0.85 0.80], 'HF');
        xlim([0 0.5]);
        xlabel('Frequency [Hz]');
        ylabel('PSD [ms^2/Hz]');
        title(sprintf('Welch PSD  (fs = %g Hz, window = %g s, overlap = %g%%)', ...
              fs, opts.windowSec, 100*opts.overlap));
        legend('PSD','VLF','LF','HF','Location','SouthWest');

        figs(end+1) = figure('Name','Poincaré','Color','w');
        plot(x, y, '.', 'MarkerSize', 6, 'Color', [0.15 0.35 0.65]); hold on; grid on; axis equal;
        % Centre and SD1/SD2 ellipse axes
        c = [mean(x), mean(y)];
        theta = linspace(0, 2*pi, 360);
        R = [cos(pi/4) -sin(pi/4); sin(pi/4) cos(pi/4)];   % +45° rotation
        E = R * [pc.SD2*cos(theta); pc.SD1*sin(theta)];
        plot(c(1)+E(1,:), c(2)+E(2,:), 'r-', 'LineWidth', 1.6);
        % Axes lines
        ax1 = R * [-pc.SD2 pc.SD2; 0 0];
        ax2 = R * [0 0; -pc.SD1 pc.SD1];
        plot(c(1)+ax1(1,:), c(2)+ax1(2,:), 'r-', 'LineWidth', 1.2);
        plot(c(1)+ax2(1,:), c(2)+ax2(2,:), 'r-', 'LineWidth', 1.2);
        % Identity line
        lims = [min([x;y]) max([x;y])];
        plot(lims, lims, 'k:', 'LineWidth', 0.8);
        xlim(lims); ylim(lims);
        xlabel('RR_n [ms]'); ylabel('RR_{n+1} [ms]');
        title(sprintf('Poincaré plot   SD1 = %.1f ms,  SD2 = %.1f ms,  SD1/SD2 = %.3f', ...
              pc.SD1, pc.SD2, pc.SD1_SD2));

        figs(end+1) = figure('Name','DFA alpha1','Color','w');
        loglog(dfa.n, dfa.F, 'o', 'MarkerFaceColor', [0.15 0.35 0.65], ...
               'MarkerEdgeColor', 'none'); hold on; grid on;
        % Fitted line
        p = polyfit(log10(dfa.n), log10(dfa.F), 1);
        nFit = [min(dfa.n) max(dfa.n)];
        loglog(nFit, 10.^polyval(p, log10(nFit)), 'r-', 'LineWidth', 1.4);
        xlabel('Scale n [beats]'); ylabel('F(n)');
        title(sprintf('DFA   \\alpha_1 = %.3f', dfa.alpha1));
        legend('F(n)', sprintf('fit, slope = %.3f', dfa.alpha1), 'Location', 'NorthWest');
    end

    %% ---------- 9. Pack results ----------
    results.timeDomain = td;
    results.freqDomain = fd;
    results.poincare   = pc;
    results.dfa        = dfa;
    results.figures    = figs;
    results.opts       = opts;
end

% ====================================================================
function opts = setDefaults(opts)
    d.fs         = 4;
    d.windowSec  = 256;
    d.overlap    = 0.5;
    d.rrMin      = 300;
    d.rrMax      = 2000;
    d.rejectFrac = 0.20;
    d.detrend    = 'linear';
    d.dfaScales  = 4:1:16;
    d.plot       = true;
    fns = fieldnames(d);
    for k = 1:numel(fns)
        if ~isfield(opts, fns{k}) || isempty(opts.(fns{k}))
            opts.(fns{k}) = d.(fns{k});
        end
    end
end

% ====================================================================
function [alpha, n, F] = computeDFA(x, scales)
% Detrended Fluctuation Analysis on a 1-D series x.
% Returns the scaling exponent alpha = d log F / d log n.

    x = x(:);
    N = numel(x);

    % Integrated profile y(k) = sum_{i=1..k} (x(i) - mean(x))
    y = cumsum(x - mean(x));

    n = scales(:)';
    F = nan(size(n));
    for k = 1:numel(n)
        nk = n(k);
        nseg = floor(N / nk);
        if nseg < 4, continue; end             % need enough segments
        Y  = reshape(y(1:nseg*nk), nk, nseg);  % each column = one window
        tt = (1:nk)';
        F2 = zeros(1, nseg);
        for s = 1:nseg
            p = polyfit(tt, Y(:,s), 1);        % linear local trend
            res = Y(:,s) - polyval(p, tt);
            F2(s) = mean(res.^2);
        end
        F(k) = sqrt(mean(F2));
    end
    valid = ~isnan(F);
    p = polyfit(log10(n(valid)), log10(F(valid)), 1);
    alpha = p(1);
    n = n(valid);
    F = F(valid);
end

% ====================================================================
function shadeBand(lo, hi, col, label)
    yl = ylim;
    patch([lo hi hi lo], [yl(1) yl(1) yl(2) yl(2)], col, ...
          'EdgeColor', 'none', 'FaceAlpha', 0.25, 'DisplayName', label);
    ylim(yl);
end
