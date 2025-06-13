import os
import json
from openai import OpenAI
from file_module.functions.reader import PdfReader
from llm.get_llm_key import get_llm_key

# 配置路径
input_folder = "input"
output_file = "output/提取结果.txt"

# 初始化客户端
client = OpenAI(
    base_url="https://api.rcouyi.com/v1",
    api_key= get_llm_key()
)


def detect_doc_type(text):
    """智能判断文档类型"""
    prompt = f"""
    请判断以下文本属于哪种文档类型：
    {text[:1000]}  # 仅分析前1000字符

    可选类型：
    - 专利（包含专利号、申请日、授权日等信息）
    - 论文（包含标题、作者、期刊、DOI等信息）
    - 其他

    只需返回最可能的类型（单字）："专"、"论"或"其"
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "你是一个文档类型分类专家"},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()


def extract_patent_info(text):
    """提取专利信息"""
    prompt = f"""
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
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "你是一个专利信息提取专家"},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)


def extract_paper_info(text):
    """提取论文信息"""
    prompt = f"""
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
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "你是一个学术论文解析专家"},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)


def process_files():
    """处理所有文件"""
    pdf_reader = PdfReader()
    results = []

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith('.pdf'):
            continue

        filepath = os.path.join(input_folder, filename)
        try:
            print(f"处理中: {filename}")
            text = pdf_reader.read(filepath)
            doc_type = detect_doc_type(text)

            if doc_type == "专":
                info = extract_patent_info(text)
                result = f"""
                    文件: {filename}
                    类型: 专利
                    专利号: {info.get('专利号', 'N/A')}
                    申请日期: {info.get('申请日期', 'N/A')}
                    授权日期: {info.get('授权日期', 'N/A')}
                    发明人: {info.get('发明人', 'N/A')}
                    受让人: {info.get('受让人', 'N/A')}
                    {"=" * 40}
                    """
            elif doc_type == "论":
                info = extract_paper_info(text)
                result = f"""
                    文件: {filename}
                    类型: 论文
                    标题: {info.get('标题', 'N/A')}
                    作者: {info.get('作者', 'N/A')}
                    期刊: {info.get('期刊', 'N/A')}
                    年份: {info.get('年份', 'N/A')}
                    DOI: {info.get('DOI', 'N/A')}
                    {"=" * 40}
                """
            else:
                result = f"""
                    文件: {filename}
                    类型: 未识别
                    {"=" * 40}
                """
            results.append(result)

        except Exception as e:
            results.append(f"\n文件: {filename}\n错误: {str(e)}\n{'=' * 40}")

    # 写入结果
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== 文档信息提取报告 ===\n")
        f.write("\n".join(results))
    print(f"处理完成！结果保存至: {output_file}")


if __name__ == "__main__":
    process_files()