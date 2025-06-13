import os
import torch
from openai import OpenAI
from transformers import BertModel, BertTokenizer, BertConfig

class ChatModel():
    def __init__(self,base_url, api_key, model_name,system_prompt, output_format):
        self.base_url = base_url
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.output_format = output_format
    def __create_message(self, user_input):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input}
        ]
        return messages

    def generate(self, user_input):
        completion = self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=self.__create_message(user_input),
            response_format=self.output_format,

        )
        return completion.choices[0].message.parsed

class OpenaiEmbeddingModel():
    def __init__(self, model_name, api_key, base_url):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, user_input):
        try:
            response = self.client.embeddings.create(
                input=user_input,
                model=self.model_name,
            )
            embedding_vector = response.data[0].embedding
            return embedding_vector
        except Exception as e:
            print(f"Error while generating embedding: {e}")
            return []

class LocalEmbeddingModel():
    def __init__(self, model_path):
        self.model = None
        self.tokenizer = None
        self.model_name = "DMetaSoul/Dmeta-embedding"  ##需要该
        self.config_path = os.path.join(model_path, 'config.json')
        self.vocab_path = os.path.join(model_path, 'vocab.txt')
        self.__load_model()
    def __load_model(self):
        config = BertConfig.from_pretrained(self.config_path)
        self.tokenizer = BertTokenizer.from_pretrained(self.vocab_path)
        self.model = BertModel.from_pretrained(self.model_name, config=config)

    def generate(self, user_input):
        try:
            inputs = self.tokenizer(user_input, return_tensors='pt', padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
            return cls_embedding.flatten().tolist()
        except Exception as e:
            print(f"Error while generating embedding: {e}")
            return []

