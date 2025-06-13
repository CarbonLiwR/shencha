import os
import torch
import asyncio
from transformers import BertModel, BertTokenizer, BertConfig
from ..base.model import BaseEmbeddingModel, BaseImageModel, BaseChatModel, BaseAudioModel


class LocalEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_path):
        super(LocalEmbeddingModel, self).__init__()
        self.model = None
        self.tokenizer = None
        self.model_loaded_event = asyncio.Event()
        self.model_name = "DMetaSoul/Dmeta-embedding"
        self.config_path = os.path.join(model_path, 'config.json')
        self.vocab_path = os.path.join(model_path, 'vocab.txt')
        self.__load_model()
    def __load_model(self):
        config = BertConfig.from_pretrained(self.config_path)
        self.tokenizer = BertTokenizer.from_pretrained(self.vocab_path)
        self.model = BertModel.from_pretrained(self.model_name, config=config)


    def generate(self, user_input):
        try:
            # Tokenize input text
            inputs = self.tokenizer(user_input, return_tensors='pt', padding=True, truncation=True, max_length=512)

            with torch.no_grad():
                outputs = self.model(**inputs)

            # Get [CLS] token vector
            cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()

            return cls_embedding.flatten().tolist()
        except Exception as e:
            print(f"Error while generating embedding: {e}")
            return []

class LocalChatModel(BaseChatModel):
    def __init__(self):
        super(LocalChatModel, self).__init__()


    def generate(self, user_input):
        pass

class LocalImageModel(BaseImageModel):
    def __init__(self):
        super(LocalImageModel, self).__init__()
        pass

    def generate(self, user_input):
        pass

class LocalAudioModel(BaseAudioModel):
    def __init__(self):
        super(LocalAudioModel, self).__init__()
        pass

    def generate(self, user_input):
        pass