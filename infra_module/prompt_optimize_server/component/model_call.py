from openai import OpenAI


class OpenaiChatModel():
    def __init__(self,base_url, api_key, model_name,system_prompt):
        super(OpenaiChatModel, self).__init__()
        self.base_url = base_url
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.system_prompt = system_prompt
    def __create_message(self, user_input):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input}
        ]
        return messages


    def generate(self, user_input):
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages = self.__create_message(user_input)
        )
        return completion.choices[0].message.content
