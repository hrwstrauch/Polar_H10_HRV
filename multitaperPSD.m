function [Pxx, f, V, lambda] = multitaperPSD(x, fs, NW, K, nfft)
% MULTITAPERPSD  Thomson multitaper one-sided PSD with locally-computed
% DPSS (Slepian) tapers.  Base MATLAB only (no Signal Processing Toolbox).
%
%   [Pxx, f] = multitaperPSD(x, fs)
%   [Pxx, f] = multitaperPSD(x, fs, NW, K, nfft)
%   [Pxx, f, V, lambda] = multitaperPSD(...)
%
%   Inputs
%     x     uniformly-sampled signal (column or row)
%     fs    sampling frequency (Hz)
%     NW    time-bandwidth product.  Default = 4.
%           Half-bandwidth W (in normalised freq.) = NW/N.
%     K     number of tapers used.  Default = 2*NW − 1.
%           The first 2*NW−1 DPSS tapers have concentration > 0.99 inside
%           [-W, W]; using more degrades resolution faster than it
%           reduces variance.
%     nfft  FFT length.  Default = 2^nextpow2(N) (zero-padded if > N).
%
%   Outputs
%     Pxx     one-sided PSD in (units of x)^2 / Hz
%     f       frequency vector
%     V       N×K matrix of DPSS tapers (each column unit L2 norm)
%     lambda  K-vector of concentration ratios in [-W, W] for each taper
%
%   Why this exists
%     For short or HRV-noisy records the VLF band has very few
%     independent samples per unit frequency, and a single Welch segment
%     gives high-variance estimates there.  Thomson's multitaper method
%     trades a small amount of bias (~1/N broadening) for a large
%     reduction in variance by averaging K orthogonal tapered periodograms.
%
%     The tapers are the discrete prolate spheroidal sequences (DPSS),
%     i.e. the eigenvectors of the band-limited concentration problem
%
%         A v = lambda v,        A_mn = sin(2 pi W (m-n)) / (pi (m-n))
%
%     The original A is numerically nasty.  Slepian (1978) showed that
%     A commutes with a simple symmetric tridiagonal matrix T whose
%     eigenvectors are the SAME as those of A, but whose spectrum is
%     well separated and well conditioned:
%
%         T_kk    = ((N-1)/2 − k)^2 · cos(2 pi W),     k = 0..N-1
%         T_k,k+1 = T_k+1,k = (k+1)(N-1-k)/2
%
%     We solve eig(T), pick the K eigenvectors corresponding to the K
%     largest eigenvalues (which map one-to-one to the K most
%     concentrated DPSS), and finally compute the actual concentration
%     ratios lambda by evaluating v' A v on the sinc-kernel matrix.
%
%   Reference
%     Thomson (1982), Proc. IEEE 70, 1055.  Slepian (1978), BSTJ 57, 1371.
%     Percival & Walden (1993), Spectral Analysis for Physical Applications.

    x = x(:);
    N = numel(x);

    if nargin < 3 || isempty(NW),   NW   = 4;                       end
    if nargin < 4 || isempty(K),    K    = max(1, 2*NW - 1);        end
    if nargin < 5 || isempty(nfft), nfft = max(2^nextpow2(N), 1024); end

    K = min(K, N);   % can't have more tapers than samples

    % -------- 1. DPSS tapers via the tridiagonal eigenvalue problem ----
    [V, lambda] = computeDPSS(N, NW, K);

    % -------- 2. Tapered periodograms ----------------------------------
    Sxx = zeros(nfft, 1);
    for k = 1:K
        Xk  = fft(V(:, k) .* x, nfft);
        Sxx = Sxx + abs(Xk).^2;
    end
    Sxx = Sxx / K;        % unweighted average (a.k.a. "low-bias" estimator)

    % -------- 3. PSD normalisation -------------------------------------
    % Each taper has unit L2 norm, so sum(V(:,k).^2) = 1.
    % The standard one-sided PSD normalisation is then |X|^2 / fs, with
    % interior bins doubled.
    Sxx = Sxx / fs;

    if mod(nfft, 2) == 0
        Kf  = nfft/2 + 1;
        Pxx = Sxx(1:Kf);
        Pxx(2:end-1) = 2 * Pxx(2:end-1);   % not DC, not Nyquist
    else
        Kf  = (nfft + 1)/2;
        Pxx = Sxx(1:Kf);
        Pxx(2:end) = 2 * Pxx(2:end);       % not DC
    end
    f = (0:Kf-1)' * (fs / nfft);
end

% =====================================================================
function [V, lambda] = computeDPSS(N, NW, K)
% Compute first K DPSS tapers of length N for time-bandwidth product NW.

    W = NW / N;     % half-bandwidth in normalised frequency

    k = (0:N-1)';
    dmain = ((N-1)/2 - k).^2 * cos(2*pi*W);
    doff  = (1:N-1)' .* (N - (1:N-1)') / 2;

    T = diag(dmain) + diag(doff, 1) + diag(doff, -1);

    [Vall, D] = eig(T);
    [~, idx]  = sort(diag(D), 'descend');
    V = Vall(:, idx(1:K));

    % Sign convention: largest-magnitude component positive
    for j = 1:K
        v = V(:, j);
        [~, im] = max(abs(v));
        if v(im) < 0
            V(:, j) = -v;
        end
    end

    % Renormalise to unit L2 (eig should already give this; be defensive)
    V = V ./ vecnorm(V);

    % -------- True concentration ratios in [-W, W] ---------------------
    % lambda_k = v_k' A v_k, where A is the sinc-kernel concentration
    % matrix.  This is what tells you which tapers are usable; the
    % eigenvalues of T are NOT the concentration ratios themselves.
    if nargout >= 2
        m  = (0:N-1)';
        % Symmetric Toeplitz sinc kernel
        c0 = 2*W;
        c  = sin(2*pi*W*m(2:end)) ./ (pi*m(2:end));
        A  = toeplitz([c0; c]);                          % N×N
        lambda = sum(V .* (A * V), 1)';                  % K×1
    end
end

% =====================================================================
function n = vecnorm(M)
% Column-wise L2 norm.  R2017b's vecnorm is implicit, but older MATLABs
% may not have it; this fallback works everywhere.
    n = sqrt(sum(M.^2, 1));
end
