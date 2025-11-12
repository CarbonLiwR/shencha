import os
import asyncio
import itertools
import aiohttp
import logging
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

logger = logging.getLogger(__name__)

# 定义全局信号量，限制并发数（根据机器性能和模型QPS设置，比如3）
semaphore = asyncio.Semaphore(3)

API_KEYS = [k.strip() for k in os.getenv("API_KEY", "").split(",") if k.strip()]
api_key_cycle = itertools.cycle(API_KEYS)

def get_next_api_key():
    return next(api_key_cycle)

# 获取配置
API_BASE_URL = os.getenv("API_BASE_URL")
TEXT_MODEL = os.getenv("TEXT_MODEL")

async def detect_doc_type(text: str) -> str:
    if not text or not text.strip():
        logger.warning("输入文本为空，跳过文档类型检测")
        return "其他"

    prompt = f"""
    分析以下文本，判断是专利、论文、标准、软著还是其他：
    {text}

    返回：专利、论文、标准、软著、其他。
    你有且只能返回上述五种类型中的一种，不要返回其他内容。
    """

    logger.info("开始检测文档类型")
    logger.debug(f"发送给大模型的提示: {prompt.strip()}")

    url = API_BASE_URL
    payload = {
        "model": TEXT_MODEL,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ],
    }

    async with semaphore:
        async with aiohttp.ClientSession() as session:
            for attempt in range(3):  # 最多重试3次
                API_KEY = get_next_api_key()  # 每次尝试都取一个Key（避免一个key被限流）
                headers = {
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json"
                }
                try:
                    async with session.post(url, json=payload, headers=headers, timeout=300) as resp:
                        data = await resp.json()

                        # --- 限流检测 ---
                        if "rate limit" in str(data).lower() or "tpm" in str(data).lower():
                            logger.warning(f"触发限流（第{attempt + 1}次），切换下一个API_KEY重试")
                            await asyncio.sleep(2 * (attempt + 1))
                            continue

                        # --- 正常响应 ---
                        if "choices" in data:
                            result = data["choices"][0]["message"]["content"].strip()
                            logger.info(f"大模型返回结果: {result}")
                            return result
                        else:
                            logger.error(f"调用模型失败（第{attempt + 1}次）: {data}")
                except Exception as e:
                    logger.warning(f"调用模型异常（第{attempt + 1}次）: {e}")
                    await asyncio.sleep(2 * (attempt + 1))

    logger.error("多次重试后仍失败，返回默认类型: 其他")
    return "其他"
