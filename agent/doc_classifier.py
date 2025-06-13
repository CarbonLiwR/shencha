import aiohttp
from llm.get_llm_key import get_llm_key


# 异步发送 POST 请求
async def send_async_request(url, headers, data):
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as response:
            if response.status == 200:
                result = await response.json()
                return result
            else:
                print(f"请求失败，状态码: {response.status}")
                print(await response.text())
                return None


# 异步文档类型检测
async def detect_doc_type(text):
    """智能判断文档类型"""
    # 初始化 OpenAI 客户端
    api_key = get_llm_key()
    # 构造请求数据
    data = {
        'model': "gpt-4o",
        'messages': [
            {
                "role": "system",
                "content": "你是一个文档类型分类专家"
            },
            {
                "role": "user",
                "content": f"""
                请判断以下文本属于哪种文档类型：
                {text[:1000]}  # 仅分析前1000字符

                可选类型：
                - 专利（包含专利号、申请日、授权日等信息）
                - 论文（包含标题、作者、期刊、DOI等信息）
                - 其他

                只需返回最可能的类型（单字）："专"、"论"或"其"
                """
            }
        ],
        'temperature': 0
    }

    # 请求 URL 和头部
    url = "https://api.rcouyi.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # 发送异步请求
    result = await send_async_request(url, headers, data)

    # 处理响应结果
    if result:
        content = result.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
        return content  # 返回单字："专"、"论"或"其"
    else:
        print("[失败] 未获取到有效响应")
        return None

