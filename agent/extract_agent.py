import os
import re
import json
import asyncio
import aiohttp
import itertools
from typing import Dict, Any
from dotenv import load_dotenv
from logging_config import logger

# ===============================
# 环境变量与全局配置
# ===============================
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")
TEXT_MODEL = os.getenv("TEXT_MODEL")

API_KEYS = [k.strip() for k in os.getenv("API_KEY", "").split(",") if k.strip()]
api_key_cycle = itertools.cycle(API_KEYS)

# 控制并发数（建议3-5，根据服务器和模型性能调整）
semaphore = asyncio.Semaphore(3)

def get_next_api_key():
    return next(api_key_cycle)


# ===============================
# 核心函数：extract_info
# ===============================
async def extract_info(text: str, doc_type: str, filename: str) -> Dict[str, Any]:
    logger.info(f"开始提取信息，文档类型: {doc_type}, 文件名: {filename}")

    # ---------- 生成 prompt ----------
    if doc_type == "专利":
        prompt = f"""
        请从以下专利文件《{filename}》的文本中提取信息：
        {text}

        返回严格 JSON 格式，包含以下字段：
        {{
          "专利号": "",
          "专利名称": "",
          "申请日期": "YYYY-MM-DD",
          "授权日期": "YYYY-MM-DD 或 N/A",
          "发明人": "以逗号分隔",
          "受让人": "公司或机构名称"
        }}
        """
    elif doc_type == "论文":
        prompt = f"""
        请从以下论文文件《{filename}》中提取信息：
        {text}

        返回严格 JSON 格式，包含：
        {{
          "标题": "",
          "作者": "张三; 李四",
          "期刊": "",
          "year": 2024,
          "DOI": "",
          "received_date": "YYYY-MM-DD",
          "accepted_date": "YYYY-MM-DD",
          "published_date": "YYYY-MM-DD",
          "project_number": "",
          "institution": ""
        }}
        没有的字段填 "N/A"。
        """
    elif doc_type == "标准":
        prompt = f"""
        请从以下标准文件《{filename}》中提取信息：
        {text}

        返回严格 JSON 格式，包含：
        {{
          "标准名称": "",
          "标准形式": "国标/地标/团标",
          "标准编号": "",
          "起草单位": "",
          "起草人": "",
          "发布单位": "",
          "发布时间": "YYYY-MM-DD",
          "实施时间": "YYYY-MM-DD"
        }}
        """
    elif doc_type == "软著":
        prompt = f"""
        请从以下软件著作权登记文件《{filename}》中提取信息：
        {text}

        返回严格 JSON 格式，包含：
        {{
          "证书号": "",
          "软件名称": "",
          "著作权人": "",
          "登记号": "",
          "授权时间": "YYYY-MM-DD"
        }}
        """
    else:
        raise ValueError(f"未知的文档类型: {doc_type}")

    # ---------- 构造请求 ----------
    url = API_BASE_URL
    payload = {
        "model": TEXT_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "你是一个信息提取专家。\n" + prompt}]
            }
        ],
    }

    # ---------- 并发控制 + 限流 + 重试 ----------
    async with semaphore:
        async with aiohttp.ClientSession() as session:
            for attempt in range(3):
                API_KEY = get_next_api_key()
                headers = {
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json"
                }

                try:
                    async with session.post(url, json=payload, headers=headers, timeout=400) as resp:
                        data = await resp.json()
                        logger.debug(f"大模型响应: {data}")

                        # --- 限流检测 ---
                        if "rate limit" in str(data).lower() or "tpm" in str(data).lower():
                            logger.warning(f"触发限流（第{attempt+1}次），切换下一个API_KEY重试")
                            await asyncio.sleep(2 * (attempt + 1))
                            continue

                        if "choices" not in data:
                            logger.error(f"调用失败（第{attempt+1}次）: {data}")
                            await asyncio.sleep(2 * (attempt + 1))
                            continue

                        content = data["choices"][0]["message"]["content"].strip()
                        return _parse_json_from_response(content)

                except Exception as e:
                    logger.warning(f"调用模型异常（第{attempt+1}次）: {e}")
                    await asyncio.sleep(2 * (attempt + 1))

    logger.error("多次重试后仍失败，返回空结果")
    return {"error": "信息提取失败"}


# ===============================
# 工具函数：安全解析JSON
# ===============================
def _parse_json_from_response(content: str) -> Dict[str, Any]:
    """
    尝试从模型返回文本中安全提取JSON
    """
    try:
        # 直接解析（如果返回就是纯JSON）
        return json.loads(content)
    except json.JSONDecodeError:
        # 提取JSON部分（兼容模型前后夹杂文字的情况）
        match = re.search(r"\{[\s\S]*\}", content)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception as e:
                logger.error(f"JSON提取失败: {e}")
        logger.error(f"未找到有效JSON，原始内容: {content[:200]}...")
        return {"error": "解析失败"}
