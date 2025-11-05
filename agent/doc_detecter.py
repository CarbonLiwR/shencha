import os
from dotenv import load_dotenv

import requests
import json
# 加载环境变量
load_dotenv()
# 获取配置
API_BASE_URL = os.getenv('API_BASE_URL')
API_KEY = os.getenv('API_KEY')
TEXT_MODEL = os.getenv('TEXT_MODEL')
VISION_MODEL = os.getenv('VISION_MODEL')
from logging_config import logger

async def detect_doc_type(text: str) -> str:
    prompt = f"""
    分析以下文本，判断是专利、论文、标准、软著还是其他：
    {text}

    返回：专利、论文、标准、软著、其他,你只能返回专利、论文、标准、软著、其他
    """

    logger.info("开始检测文档类型")
    logger.info(f"发送给大模型的提示: {prompt}")

    url = API_BASE_URL

    payload = {
        "model": TEXT_MODEL,
        "messages": [{"role": "user", "content": prompt}]
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        print(response.status_code)
        print(response.text)
        result = response.text

        logger.info(f"大模型返回结果: {result}")
        return result
    except Exception as e:
        logger.error(f"调用大模型失败: {str(e)}", exc_info=True)
        return "其他"
