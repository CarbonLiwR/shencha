import json,re,os
from typing import Dict, Any
from dotenv import load_dotenv
import requests
# 加载环境变量
load_dotenv()
# 获取配置
API_BASE_URL = os.getenv('API_BASE_URL')
API_KEY = os.getenv('API_KEY')
TEXT_MODEL = os.getenv('TEXT_MODEL')
VISION_MODEL = os.getenv('VISION_MODEL')


async def extract_info(text: str, doc_type: str, filename: str) -> Dict[str, Any]:
    first = text[:12000] # 限制前12000个字符
    last = text[-12000:] # 限制后12000个字符
    print(f"提取信息，文档类型: {doc_type}, 文件名: {filename}")
    if doc_type == '专利':
        prompt = f"""
        从以下专利文件{filename}文本中提取信息：
        {first}

        要求返回严格JSON格式，请提取以下字段：
        1. 专利号
        2. 专利名称
        3. 申请日期（YYYY-MM-DD）
        4. 授权日期（如无则写N/A）
        5. 发明人（逗号分隔）
        6. 受让人（公司/机构）

        返回 JSON 格式。
        """
    elif doc_type == '论文':
        prompt = f"""
        请从以下论文文件{filename}文本中精确提取信息：
        {first,last}

        要求返回严格JSON格式，包含以下字段：
        1. 标题（必须提取）
        2. 作者（分号分隔，如"张三; 李四; 王五"）
        3. 期刊/会议名称（完整名称）
        4. 发表年份（YYYY，必须从文本中提取）
        5. DOI（完整格式，如"10.1002/ajh.27272"，若无则写N/A）
        6. received_date（收稿日期，YYYY-MM-DD格式）
        7. accepted_date（接受日期，YYYY-MM-DD格式）
        8. published_date（出版日期，YYYY-MM-DD格式）
        9. project_number（项目编号，若无则写N/A）
        10. institution（单位/机构，若无则写N/A）

        特别注意：
        - 日期格式示例：Received:4December2023 → received_date: "2023-12-04"
        - 必须包含所有8个字段，没有的字段写N/A
        - 年份优先从出版日期提取，其次接受日期，最后收稿日期
        - 如果可行的话，项目编号需要与对应的资助机构一起提供

        示例格式：
        {{
          "标题": "Report of IRF2BP1 as a novel partner of RARA in variant acute promyelocytic leukemia",
          "作者": "Jiang Bin; Zhang San; Li Si",
          "期刊": "American Journal of Hematology",
          "year": 2024,
          "DOI": "10.1002/ajh.27272",
          "received_date": "2023-12-04",
          "accepted_date": "2024-02-18",
          "published_date": "2024-03-01"
          "project_number": "国家自然科学基金No.12345678",
          "institution": "北京大学医学部"
        }}
        """
    elif doc_type == '标准':
        prompt = f"""
        请从以下标准文件{filename}文本中精确提取信息：
        {first}

        要求返回严格JSON格式，包含以下字段：
        1. 标准名称（完整名称）
        2. 标准形式（如 国标 地标 团标）
        2. 标准编号（如GB/T 12345-2020）
        3. 起草单位（分号分隔）
        4. 起草人（分号分隔）
        5. 发布单位
        6. 发布时间（YYYY-MM-DD格式）
        7. 实施时间（YYYY-MM-DD格式）

        特别注意：
        - 必须包含所有7个字段，没有的字段写N/A
        - 日期格式必须统一为YYYY-MM-DD

        示例格式：
        {{
          "标准名称": "信息技术 软件产品评价 质量特性及其使用指南",
          "标准编号": "GB/T 16260-2006",
          "起草单位": "中国电子技术标准化研究所;北京大学",
          "起草人": "张三;李四;王五",
          "发布单位": "中华人民共和国国家质量监督检验检疫总局",
          "发布时间": "2006-03-14",
          "实施时间": "2006-10-01"
        }}
        """

    elif doc_type == '软著':
        prompt = f"""
        请从以下软件著作权登记文件{filename}文本中精确提取信息（包括正文与OCR识别部分）：
        {first}

        要求返回严格JSON格式，包含以下字段：
        1. 证书号
        2. 软件名称
        3. 著作权人
        4. 登记号
        5. 授权时间（YYYY-MM-DD格式）

        特别注意：
        - 必须包含所有5个字段，没有的字段写N/A
        - 日期必须统一为YYYY-MM-DD
        - 如果文本与OCR信息冲突，以OCR信息为准

        示例格式：
        {{
          "证书号": "软著登字第1234567号",
          "软件名称": "智能语音识别系统V1.0",
          "著作权人": "北京某某科技有限公司",
          "登记号": "2024SR1234567",
          "授权时间": "2024-05-20"
        }}
        """


    else:
        raise ValueError(f"未知的文档类型: {doc_type}")

    import requests

    url = API_BASE_URL

    payload = {
        "model": TEXT_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "你是一个信息提取专家"+prompt+"你返回的JSON格式的字段必须严格按照要求，必须为中文字段名"
                    }

                ]
            },

        ]
    }
    headers = {
        "Authorization":  f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)
    response_data = response.json()
    content=response_data['choices'][0]['message']['content']
    print("response", response_data['choices'][0]['message']['content'])
    # return response_data['choices'][0]['message']['content'].strip()
    #

    try:
        result = json.loads(content)  # 尝试直接解析JSON
    except json.JSONDecodeError:
        # 方法2：如果失败，尝试提取JSON部分
        match = re.search(r"\{.*\}", content, re.DOTALL)  # 对content字符串操作
        if not match:
            raise ValueError(f"未找到JSON内容: {content}")
        result = json.loads(match.group(0))
    print("最终 result:", result)

    # 确保所有字段存在
    if doc_type == '论文':
        required_fields = [
            '标题', '作者', '期刊', 'year',
            'DOI', 'received_date', 'accepted_date', 'published_date'
        ]
        for field in required_fields:
            if field not in result:
                result[field] = "N/A"

    return result
