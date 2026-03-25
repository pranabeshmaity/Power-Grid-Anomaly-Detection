import numpy as np


class PhysicsFeatureExtractor:
    """
    Physics-informed feature extraction for power systems.
    """

    def __init__(self):
        pass

    def extract(self, windows: np.ndarray) -> np.ndarray:
        features = []

        for w in windows:
            # Frequency derivatives
            d1 = np.diff(w)
            d2 = np.diff(w, 2)

            # ROCOF (Rate of Change of Frequency)
            rocof = np.max(np.abs(d1))

            # Frequency deviation
            freq_dev = np.std(d1)

            # Inertia proxy (second derivative energy)
            inertia = np.sum(np.abs(d2))

            # Damping proxy
            damping = np.mean(np.abs(d1))

            # Stability indicator
            stability = np.var(w)

            features.append([
                rocof,
                freq_dev,
                inertia,
                damping,
                stability
            ])

        return np.array(features)