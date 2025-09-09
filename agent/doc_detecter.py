import os
import sys
from config.llm_config import llm_config
from llm.send_request import send_async_request

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
async def detect_doc_type(text: str) -> str:
    prompt = f"""
    分析以下文本，判断是专利、论文、标准、软著还是其他：
    {text[:1000]}

    返回：专利、论文、标准、软著、其他,你只能返回专利、论文、标准、软著、其他
    """
    headers = {
        "Content-Type": "application/json",
    }
    if llm_config.api_key and llm_config.api_key.strip():
        headers["Authorization"] = f"Bearer {llm_config.api_key}"
    else:
        print("API 密钥为空，将不使用 Authorization 头部。")
    
    data = {
        'model': llm_config.model_name,
        'messages': [
            {"role": "system", "content": "你是一个文档分类专家"},
            {"role": "user", "content": prompt}
        ],
    }
    url=llm_config.api_url
    response = await send_async_request(url, headers, data)
    # print("response", response['choices'][0]['message']['content'])
    return response['choices'][0]['message']['content'].strip()
