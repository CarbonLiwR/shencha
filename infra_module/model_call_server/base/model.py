from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self):
        pass

    def generate(self, user_input):
        pass

class BaseChatModel(BaseModel):
    def __init__(self):
        super(BaseChatModel, self).__init__()

    def generate(self, user_input):
        pass

class BaseAudioModel(BaseModel):
    def __init__(self):
        super(BaseAudioModel, self).__init__()
        pass

    def generate(self, user_input):
        pass

class BaseEmbeddingModel(BaseModel):
    def __init__(self):
        super(BaseEmbeddingModel, self).__init__()

    def generate(self, user_input):
        pass

class BaseImageModel(BaseModel):
    def __init__(self):
        super(BaseImageModel, self).__init__()

    def generate(self, user_input):
        pass