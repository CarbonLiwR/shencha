import logging
import openai
from openai import OpenAI

class OpenAIHandle:
    def __init__(self,key,url):
        self.key = key
        self.url = url
    def answer_llm(self, system_content, user_content):
        messages = [
            {"role": "system", "content": system_content },
            {
                "role": "user",
                "content": user_content
            }
        ]
        try:
            client = OpenAI(api_key=self.key, base_url=self.url)
            resp = client.chat.completions.create(
                model="gpt-4o",
                temperature = 0.3,
                messages=messages
            )
            return str(resp.choices[0].message.content)
        except openai.APIError as e:
            logging.error(f"API错误: {e.message}")
            raise