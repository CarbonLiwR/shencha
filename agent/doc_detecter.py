import aiohttp
from imageio.v3 import improps
from openai import AsyncOpenAI
from llm.get_llm_key import get_llm_key
import asyncio


client = AsyncOpenAI(
    base_url="https://api.rcouyi.com/v1",
    api_key="sk-pAauG9ss64pQW9FVA703F1453b334eFb95B7447b9083BaBd"
)

async def detect_doc_type(text: str) -> str:
    prompt = f"""
    分析以下文本，判断是专利、论文还是其他：
    {text[:1000]}

    返回：专利、论文、其他,你只能返回专利、论文、其他
    """
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "你是一个文档分类专家"},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    print("response", response.choices[0].message.content)
    return response.choices[0].message.content.strip()

