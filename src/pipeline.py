import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class OscillationPipeline:
    """
    Classical ML pipeline using Isolation Forest with feature scaling.
    """

    def __init__(self, contamination=0.05):
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42
        )

    def detect(self, features: np.ndarray) -> np.ndarray:
        # Normalize features
        features_scaled = self.scaler.fit_transform(features)

        # Train model
        self.model.fit(features_scaled)

        # Predict anomalies
        preds = self.model.predict(features_scaled)

        return np.where(preds == -1)[0]