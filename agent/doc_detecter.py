import os
import sys
from dotenv import load_dotenv
import requests
# 加载环境变量
load_dotenv()
# 获取配置
API_BASE_URL = os.getenv('API_BASE_URL')
API_KEY = os.getenv('API_KEY')
TEXT_MODEL = os.getenv('TEXT_MODEL')
VISION_MODEL = os.getenv('VISION_MODEL')


sys.path.append(os.path.dirname(os.path.dirname(__file__)))
async def detect_doc_type(text: str) -> str:
    prompt = f"""
    分析以下文本，判断是专利、论文、标准、软著还是其他：
    {text[:1000]}

    返回：专利、论文、标准、软著、其他,你只能返回专利、论文、标准、软著、其他
    """


    url = API_BASE_URL

    payload = {
        "model": TEXT_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }

                ]
            },

        ]
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.post(url, json=payload, headers=headers)
    response_data = response.json()
    print("response", response_data['choices'][0]['message']['content'])
    return response_data['choices'][0]['message']['content'].strip()
