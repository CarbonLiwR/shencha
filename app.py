import os
from typing import List
import tempfile
import shutil
from file_module.functions.reader import PdfReader
from fastapi import FastAPI, HTTPException,UploadFile,File
from pydantic import BaseModel, HttpUrl
from agent.extract_agent import extract_paper_info,extract_patent_info
from agent.doc_classifier import detect_doc_type
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv


load_dotenv()

# FastAPI 应用实例
app = FastAPI(
    title="文档信息提取服务",
    description="提供文档信息提取服务的API接口",
    version="1.0.0"
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 定义响应模型
class ProcessResponse(BaseModel):
    results: List[str]  # 处理后的结果列表

# 异步处理上传文件的主逻辑
@app.post("/api/v1/process_files", response_model=ProcessResponse)
async def process_files(files: List[UploadFile] = File(...)):
    """处理上传的文档并提取信息"""
    pdf_reader = PdfReader()  # 假设 PdfReader 是一个可以读取 PDF 文件的类
    results = []

    # 创建临时目录
    temp_dir = tempfile.mkdtemp()

    try:
        # 保存上传的文件到临时目录
        for file in files:
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

        # 处理临时目录中的所有文件
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            if not filename.lower().endswith('.pdf'):
                results.append(f"文件: {filename}\n错误: 非PDF文件\n{'=' * 40}")
                continue

            try:
                print(f"处理中: {filename}")
                text = pdf_reader.read(file_path)  # 读取PDF内容
                doc_type = await detect_doc_type(text)  # 异步检测文档类型

                if doc_type == "专":
                    info = await extract_patent_info(text)  # 异步提取专利信息
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
                    info = await extract_paper_info(text)  # 异步提取论文信息
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
        with open("output/提取结果.txt", 'w', encoding='utf-8') as f:
            f.write("=== 文档信息提取报告 ===\n")
            f.write("\n".join(results))
        print(f"处理完成！结果保存至: output/提取结果.txt")
    finally:
        # 清理临时目录
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass

    return ProcessResponse(results=results)

# 启动服务器
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
