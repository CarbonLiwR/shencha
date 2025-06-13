from openai import OpenAI
from ..base.model import BaseEmbeddingModel, BaseImageModel, BaseChatModel, BaseAudioModel

class OpenaiEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name, api_key, base_url):
        super(OpenaiEmbeddingModel, self).__init__()
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

class OpenaiChatModel(BaseChatModel):
    def __init__(self,base_url, api_key, model_name,system_prompt, output_format):
        super(OpenaiChatModel, self).__init__()
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

class OpenaiImageModel(BaseImageModel):
    def __init__(self):
        super(OpenaiImageModel, self).__init__()
        pass

    def generate(self, user_input):
        pass

class OpenaiAudioModel(BaseAudioModel):
    def __init__(self):
        super(OpenaiAudioModel, self).__init__()
        pass

    def generate(self, user_input):
        pass