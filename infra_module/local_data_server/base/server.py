
from abc import ABC, abstractmethod


class LocalDataServer(ABC):
    def __init__(self):
        self.data_semantic_retriever = None
        self.field_ranker = None
        self.regex_chunker = None
        self.data_loader = None

