from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import os
import json
import tempfile
import shutil
from file_module.functions.reader import PdfReader
from openai import AsyncOpenAI  # 修改导入
import pdfplumber

# 加载环境变量（如果有）
from dotenv import load_dotenv

from llm.get_llm_key import get_llm_key

load_dotenv()

app = FastAPI(title="文档信息提取服务", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化 OpenAI 客户端
client = AsyncOpenAI(
    base_url="https://api.rcouyi.com/v1",
    api_key=get_llm_key()
)

class ValidityCheckRequest(BaseModel):
    start_date: str = Field(..., description="起始日期 (YYYY-MM-DD)")
    end_date: str = Field(..., description="结束日期 (YYYY-MM-DD)")
    docs: List[Dict[str, Any]] = Field(..., description="已提取的文档结构化信息")

class ValidityCheckResponse(BaseModel):
    valid_docs: List[dict]
    total_valid: int
    time_range: str

class ProcessResponse(BaseModel):
    results: List[str]
    data: List[dict]


def parse_date(date_str: str) -> Optional[datetime]:
    formats = [
        "%Y-%m-%d", "%Y/%m/%d", "%Y年%m月%d日",
        "%Y.%m.%d", "%d-%m-%Y", "%d/%m/%Y",
        "%b %d, %Y", "%B %d, %Y"
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None

def check_validity(item: dict, start_date: str, end_date: str) -> bool:
    try:
        start_dt = parse_date(start_date)
        end_dt = parse_date(end_date)
        if not all([start_dt, end_dt]):
            return False

        if item.get('类型') == '专利':
            date_str = item.get('授权日期')
            if date_str == 'N/A' or not date_str:  # 如果授权日期为 'N/A' 或为空
                date_str = item.get('申请日期', 'N/A')
        else:
            year = item.get('年份')
            date_str = f"{year}-01-01" if year and year != 'N/A' else None

        doc_date = parse_date(date_str)
        if not doc_date:
            return False
        return start_dt <= doc_date <= end_dt
    except:
        return False

async def extract_info(text: str, doc_type: str) -> Dict[str, Any]:
    if doc_type == '专利':
        prompt = f"""
        从以下专利文本中提取信息：
        {text[:5000]}

        请提取：
        1. 专利号
        2. 申请日期（YYYY-MM-DD）
        3. 授权日期（如无则写N/A）
        4. 发明人（逗号分隔）
        5. 受让人（公司/机构）

        返回 JSON 格式。
        """
    else:
        prompt = f"""
        从以下论文文本中提取信息：
        {text[:5000]}

        请提取：
        1. 标题
        2. 作者（分号分隔）
        3. 期刊/会议名称
        4. 发表年份（YYYY）
        5. DOI（如无则写N/A）

        返回 JSON 格式。
        """

    role = "你是一个信息提取专家"

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": role},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

async def detect_doc_type(text: str) -> str:
    prompt = f"""
    分析以下文本，判断是专利、论文还是其他：
    {text[:1000]}

    返回：专利、论文、其他
    """
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "你是一个文档分类专家"},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()

@app.post("/api/v1/process_files", response_model=ProcessResponse)
async def process_files(files: List[UploadFile] = File(...)):
    pdf_reader = PdfReader()
    temp_dir = tempfile.mkdtemp()
    results = []
    structured_data = []

    try:
        for file in files:
            file_path = os.path.join(temp_dir, file.filename)
            print(f"处理中: {file.filename}")
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            # text = pdf_reader.read(file_path)

            with pdfplumber.open(file_path) as pdf:
                all_text = ""
                for page in pdf.pages:
                    text0 = page.extract_text()
                    all_text += text0 + "\n"
            text = all_text

            doc_type = await detect_doc_type(text)

            if doc_type == "专利":
                info = await extract_info(text, "专利")
                info.update({"文件名": file.filename, "类型": "专利"})
                structured_data.append(info)
                result = f"文件: {file.filename}\n类型: 专利\n专利号: {info.get('专利号')}\n申请日期: {info.get('申请日期')}\n授权日期: {info.get('授权日期')}\n发明人: {info.get('发明人')}\n受让人: {info.get('受让人')}\n{'='*40}"

            elif doc_type == "论文":
                info = await extract_info(text, "论文")
                info.update({"文件名": file.filename, "类型": "论文"})
                structured_data.append(info)
                result = f"文件: {file.filename}\n类型: 论文\n标题: {info.get('标题')}\n作者: {info.get('作者')}\n期刊: {info.get('期刊')}\n年份: {info.get('年份')}\nDOI: {info.get('DOI')}\n{'='*40}"

            else:
                # result = f"文件: {file.filename}\n类型: 未识别\n{'='*40}"
                info = await extract_info(text, "专利")
                info.update({"文件名": file.filename, "类型": "专利"})
                structured_data.append(info)
                result = f"文件: {file.filename}\n类型: 专利\n专利号: {info.get('专利号')}\n申请日期: {info.get('申请日期')}\n授权日期: {info.get('授权日期')}\n发明人: {info.get('发明人')}\n受让人: {info.get('受让人')}\n{'=' * 40}"

            results.append(result)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return ProcessResponse(results=results, data=structured_data)

@app.post("/api/v1/check_validity", response_model=ValidityCheckResponse)
async def check_documents_validity(request: ValidityCheckRequest):
    start_dt = parse_date(request.start_date)
    end_dt = parse_date(request.end_date)
    if not all([start_dt, end_dt]):
        raise HTTPException(status_code=400, detail="日期格式错误")
    if start_dt > end_dt:
        raise HTTPException(status_code=400, detail="起始日期不能晚于结束日期")

    valid_docs = [doc for doc in request.docs if check_validity(doc, request.start_date, request.end_date)]

    return ValidityCheckResponse(
        valid_docs=valid_docs,
        total_valid=len(valid_docs),
        time_range=f"{request.start_date} 至 {request.end_date}"
    )

# 启动服务器
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)