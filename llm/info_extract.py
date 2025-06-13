import aiohttp
import json
# 从文件中读取 system prompt


# 异步发送 POST 请求
from llm.get_llm_key import get_llm_key


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

# 单个请求的任务
async def info_extract(info: str, prompt):
    # 初始化 OpenAI 客户端
    api_key = get_llm_key()
    # 请求数据
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
                {info[:5000]}

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

    # 输出结果
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



# async def main():
#     # 从文件中读取 ocr info
#     ocr_info_path = 'ocr_output.txt'
#     try:
#         with open(ocr_info_path, 'r', encoding='utf-8') as f:
#             ocr_info = f.read()
#     except FileNotFoundError:
#         print(f"[错误] 找不到 ocr info 文件：{ocr_info_path}")
#         return
#
#     # 提交请求
#     result = await info_extract(ocr_info)




# 运行异步主函数
# if __name__ == "__main__":
#     try:
#         asyncio.run(main())
#     except RuntimeError:
#         # 防止在某些环境中 asyncio 冲突
#         loop = asyncio.get_event_loop()
#         loop.run_until_complete(main())


