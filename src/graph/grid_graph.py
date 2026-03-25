import networkx as nx
import numpy as np


class GridGraph:
    """
    Builds graph from PSSE topology
    """

    def __init__(self):
        self.graph = nx.Graph()

    def build(self, buses, lines):
        for b in buses:
            self.graph.add_node(b)

        for u, v in lines:
            self.graph.add_edge(u, v)

        return self.graph

    def adjacency_matrix(self):
        return nx.to_numpy_array(self.graph)

    def graph_features(self):
        degrees = dict(self.graph.degree())
        clustering = nx.clustering(self.graph)

        features = []

        for node in self.graph.nodes():
            features.append([
                degrees[node],
                clustering[node]
            ])

        return np.array(features)