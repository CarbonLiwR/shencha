import json
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


async def extract_patent_info(text):
    """提取专利信息"""
    # 初始化 OpenAI 客户端
    api_key = get_llm_key()
    # 构造请求数据
    data = {
        'model': "gpt-4o",
        'response_format': {"type": "json_object"},
        'messages': [
            {
                "role": "system",
                "content": "你是一个专利信息提取专家"
            },
            {
                "role": "user",
                "content": f"""
                从以下专利文本中提取信息：
                {text[:5000]}

                需要提取：
                1. 专利号（如CN123456/US123456等）
                2. 申请日期（格式：YYYY-MM-DD）
                3. 授权日期（格式：YYYY-MM-DD）
                4. 发明人（多个用逗号分隔）
                5. 受让人（公司/机构）

                要求：
                - 不存在则写"N/A"
                - 用JSON返回，字段名如下：
                {{
                    "专利号": "",
                    "申请日期": "",
                    "授权日期": "",
                    "发明人": "",
                    "受让人": ""
                }}
                """
            }
        ]
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
        content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            print("[失败] 响应内容无法解析为JSON")
            return None
    else:
        print("[失败] 未获取到有效响应")
        return None



async def extract_paper_info(text):
    """提取论文信息"""
    # 初始化 OpenAI 客户端
    api_key = get_llm_key()
    # 构造请求数据
    data = {
        'model': "gpt-4o",
        'response_format': {"type": "json_object"},
        'messages': [
            {
                "role": "system",
                "content": "你是一个学术论文解析专家"
            },
            {
                "role": "user",
                "content": f"""
                从以下论文文本中提取元数据：
                {text[:5000]}

                需要提取：
                1. 标题
                2. 作者（多个用逗号分隔）
                3. 期刊/会议名称
                4. 发表年份
                5. DOI号

                要求同上，JSON字段名如下：
                {{
                    "标题": "",
                    "作者": "",
                    "期刊": "",
                    "年份": "",
                    "DOI": ""
                }}
                """
            }
        ]
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
        content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            print("[失败] 响应内容无法解析为JSON")
            return None
    else:
        print("[失败] 未获取到有效响应")
        return None


