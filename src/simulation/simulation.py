import numpy as np


class SimulationAugmentor:
    """
    Simulates realistic disturbances in power grid signals.
    """

    def __init__(self, noise_std=0.01):
        self.noise_std = noise_std

    def inject_fault(self, signal: np.ndarray) -> np.ndarray:
        """
        Add localized disturbance.
        """
        noisy = signal.copy()

        idx = np.random.randint(50, len(signal) - 50)

        # Disturbance types
        disturbance_type = np.random.choice(["spike", "drift", "oscillation"])

        if disturbance_type == "spike":
            noisy[idx:idx+10] += np.random.normal(0, self.noise_std * 5, 10)

        elif disturbance_type == "drift":
            noisy[idx:idx+20] += np.linspace(0, self.noise_std * 3, 20)

        elif disturbance_type == "oscillation":
            t = np.arange(20)
            noisy[idx:idx+20] += 0.01 * np.sin(0.5 * t)

        return noisy

    def augment(self, signal: np.ndarray) -> np.ndarray:
        """
        Generate augmented signal.
        """
        return self.inject_fault(signal)