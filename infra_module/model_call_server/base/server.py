


from abc import ABC, abstractmethod


class ModelCallServer(ABC):
    def __init__(self):
        self.openai_embedding_model = None
        self.openai_chat_model = None
        self.local_embedding_model = None


