import numpy as np


class GraphFeatureExtractor:
    """
    Graph-based feature extractor using correlation structure.
    Acts as a lightweight GNN proxy.
    """

    def __init__(self, eps=1e-8):
        self.eps = eps

    def compute_adjacency(self, windows: np.ndarray) -> np.ndarray:
        """
        Compute correlation-based adjacency matrix.
        """
        corr = np.corrcoef(windows)
        corr = np.nan_to_num(corr)
        return corr

    def compute_graph_features(self, windows: np.ndarray) -> np.ndarray:
        """
        Extract node-level graph features.
        """
        adj = self.compute_adjacency(windows)

        # Degree (connectivity strength)
        degree = np.sum(adj, axis=1)

        # Clustering proxy
        clustering = np.mean(adj, axis=1)

        # Graph energy
        energy = np.sum(adj ** 2, axis=1)

        return np.vstack([degree, clustering, energy]).T

    def transform(self, windows: np.ndarray) -> np.ndarray:
        return self.compute_graph_features(windows)