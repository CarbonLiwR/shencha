from ..datamodel.graph import Graph
from abc import ABC, abstractmethod


class KnowledgeGraphServer(ABC):
    def __init__(self):
        self.graph_retriever = None
        self.graph_loader = None

    def retrieve_graph(self, query: str, graph: Graph) -> dict:
        return self.graph_retriever.retrieve(query, graph)

    def load_graph_data(self, data_path) -> Graph:
        return self.graph_loader.load(data_path)
