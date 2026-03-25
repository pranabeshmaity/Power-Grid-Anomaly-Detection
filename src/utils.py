import numpy as np


# =========================
# WINDOWING
# =========================
def create_windows(signal: np.ndarray, window_size: int) -> np.ndarray:
    """
    Convert 1D signal into overlapping windows.
    """
    if len(signal) <= window_size:
        raise ValueError("Signal length must be greater than window size")

    return np.array([
        signal[i:i + window_size]
        for i in range(len(signal) - window_size)
    ])


# =========================
# GRAPH STRUCTURE (GNN-STYLE)
# =========================
def compute_graph_structure(windows: np.ndarray) -> np.ndarray:
    """
    Correlation-based adjacency matrix.
    Acts as graph structure between windows.
    """
    corr = np.corrcoef(windows)
    corr = np.nan_to_num(corr)  # handle NaNs safely
    return corr


# =========================
# SIMULATION AUGMENTATION
# =========================
def simulate_disturbance(signal: np.ndarray) -> np.ndarray:
    """
    Inject synthetic disturbance (fault-like behavior).
    """
    if len(signal) < 100:
        return signal.copy()

    noisy = signal.copy()

    idx = np.random.randint(50, len(signal) - 50)

    # disturbance window
    length = 20

    # Gaussian spike
    noise = np.random.normal(0, 0.01, length)

    noisy[idx:idx + length] += noise

    return noisy


# =========================
# FEATURE ENGINEERING
# =========================
def extract_features(windows: np.ndarray) -> np.ndarray:
    """
    Multi-domain feature extraction:
    - Time-domain
    - Frequency-domain
    - Physics-informed features
    """

    features = []

    for w in windows:
        w = np.asarray(w)

        # =========================
        # TIME DOMAIN
        # =========================
        mean = np.mean(w)
        std = np.std(w)
        max_val = np.max(w)
        min_val = np.min(w)

        # =========================
        # TREND
        # =========================
        slope = np.polyfit(np.arange(len(w)), w, 1)[0]

        # =========================
        # ENERGY
        # =========================
        energy = np.sum(w ** 2)

        # =========================
        # OSCILLATION
        # =========================
        zero_cross = np.sum(np.diff(np.sign(w)) != 0)

        # =========================
        # FREQUENCY DOMAIN
        # =========================
        fft_vals = np.abs(np.fft.rfft(w))  # more stable than fft
        dominant_freq = np.argmax(fft_vals)

        psd = fft_vals ** 2
        psd_norm = psd / (np.sum(psd) + 1e-8)
        spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-8))

        # =========================
        # TEMPORAL STRUCTURE
        # =========================
        autocorr = np.correlate(w, w, mode='full')[len(w) - 1]

        # =========================
        # PHYSICS-INFORMED FEATURES
        # =========================
        d1 = np.diff(w)
        d2 = np.diff(w, 2)

        rocof = np.max(np.abs(d1)) if len(d1) > 0 else 0
        freq_dev = np.std(d1) if len(d1) > 0 else 0
        voltage_var = np.var(w)

        inertia_proxy = np.sum(np.abs(d2)) if len(d2) > 0 else 0
        damping_proxy = np.mean(np.abs(d1)) if len(d1) > 0 else 0

        features.append([
            mean, std, max_val, min_val,
            slope, energy, zero_cross,
            dominant_freq,
            autocorr, spectral_entropy,
            rocof, freq_dev, voltage_var,
            inertia_proxy, damping_proxy
        ])

    return np.array(features)