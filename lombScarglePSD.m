function [Pxx, f] = lombScarglePSD(t, x, fmin, fmax, df)
% LOMBSCARGLEPSD  Lomb–Scargle power spectral density for unevenly
% sampled HRV.  Base MATLAB only (no Signal Processing Toolbox).
%
%   [Pxx, f] = lombScarglePSD(t, x, fmin, fmax, df)
%
%   Inputs
%     t     sample times in seconds (column or row), monotonically
%           increasing, NOT required to be uniformly spaced.
%     x     sample values in the same order (e.g., NN intervals in ms).
%     fmin  lowest analysis frequency in Hz.  Default = 1/T_total.
%     fmax  highest analysis frequency in Hz. Default = 0.5  (HF upper edge).
%     df    frequency-grid resolution in Hz.  Default = 1/(4*T_total),
%           i.e. 4x oversampling of the natural Rayleigh resolution.
%
%   Outputs
%     Pxx   one-sided power spectral density in (units of x)^2 / Hz,
%           normalised so that  trapz(f, Pxx) ≈ var(x).
%     f     frequency vector (column).
%
%   Why this exists
%     The Welch pipeline in hrv_analysis.m has to resample the irregular
%     beat-to-beat NN series onto a uniform 4 Hz grid via pchip.  That
%     interpolation step distorts the spectrum, particularly for short
%     records or records with many rejected beats — the interpolant fills
%     in beats that were not measured, biasing HF power upward and
%     fabricating spectral content.  The Lomb–Scargle periodogram is the
%     least-squares fit of a single sinusoid (A cos + B sin) at each
%     trial frequency f, evaluated directly on the irregular (t_i, x_i)
%     pairs.  No interpolation, no aliasing introduced by resampling.
%     The time shift tau(f) below makes the cos and sin components exactly
%     orthogonal on the irregular grid (Scargle 1982), so the resulting
%     statistic has the same distribution as a classical periodogram.
%
%   Reference
%     Scargle (1982), ApJ 263, 835.  Press & Rybicki (1989), ApJ 338, 277.
%     Laguna et al. (1998), IEEE TBME 45, 698 — HRV-specific evaluation.

    t = t(:);  x = x(:);
    N = numel(x);
    T_total = t(end) - t(1);

    if nargin < 3 || isempty(fmin), fmin = 1/T_total;       end
    if nargin < 4 || isempty(fmax), fmax = 0.5;             end
    if nargin < 5 || isempty(df),   df   = 1/(4*T_total);   end

    % Centre the series so the periodogram is on fluctuations only
    xc = x - mean(x);

    f  = (fmin:df:fmax)';
    nf = numel(f);
    P  = zeros(nf, 1);              % "classical" Lomb–Scargle in units of x^2

    for k = 1:nf
        omega = 2*pi*f(k);

        % Time shift tau(f) for orthogonality (Scargle 1982)
        s2 = sum(sin(2*omega*t));
        c2 = sum(cos(2*omega*t));
        tau = atan2(s2, c2) / (2*omega);

        wt  = omega * (t - tau);
        cwt = cos(wt);
        swt = sin(wt);

        Sxc = sum(xc .* cwt);
        Sxs = sum(xc .* swt);
        Scc = sum(cwt.^2);
        Sss = sum(swt.^2);

        P(k) = 0.5 * (Sxc.^2 ./ Scc + Sxs.^2 ./ Sss);   % units of x^2
    end

    % Convert classical Scargle periodogram to a one-sided PSD whose
    % integral over [0, fs_avg/2] is the signal variance:
    %   trapz(f, Pxx) ≈ var(x).
    % The factor 2/fs_avg comes from the same Parseval calculation that
    % gives Pxx = 2*|X|^2/(fs*N) for a one-sided Welch with rect window.
    fs_avg = (N - 1) / T_total;     % mean sampling rate (Hz)
    Pxx = (2 / fs_avg) * P;
end
