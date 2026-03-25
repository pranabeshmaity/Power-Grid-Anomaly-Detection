import numpy as np


class PowerSystemDynamics:
    """
    Physics-inspired dynamic constraints
    """

    def __init__(self):
        pass

    def swing_equation_features(self, signal):
        """
        Approximate swing equation behavior
        d2θ/dt2 = (Pm - Pe) / M
        """
        d1 = np.diff(signal)
        d2 = np.diff(signal, 2)

        inertia = np.mean(np.abs(d2))
        damping = np.mean(np.abs(d1))

        return inertia, damping

    def extract(self, windows):
        features = []

        for w in windows:
            inertia, damping = self.swing_equation_features(w)
            features.append([inertia, damping])

        return np.array(features)