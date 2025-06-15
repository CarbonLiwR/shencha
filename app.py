import shutil
import tempfile
from datetime import datetime
from typing import List, Dict, Any, Optional

# 加载环境变量（如果有）
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent.doc_detecter import detect_doc_type
from agent.extract_agent import extract_info
from agent.pdf_reader import pdf_text_reader, pdf_pic_reader

load_dotenv()
app = FastAPI(title="文档信息提取服务", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ValidityCheckRequest(BaseModel):
    start_date: str = Field(..., description="起始日期 (YYYY-MM-DD)")
    end_date: str = Field(..., description="结束日期 (YYYY-MM-DD)")
    docs: Dict[str, Dict[str, Dict[str, Any]]] = Field(..., description="已提取的文档结构化信息，包括专利和论文")

class ValidityCheckResponse(BaseModel):
    valid_patents: List[Dict[str, Any]] = Field(..., description="有效的专利文档")
    valid_papers: List[Dict[str, Any]] = Field(..., description="有效的论文文档")
    total_valid: int = Field(..., description="有效文档总数")
    time_range: str = Field(..., description="时间范围")
    date_comparisons: List[Dict[str, Any]] = Field(..., description="日期比较结果")
    comparison_stats: Dict[str, Dict[str, int]] = Field(..., description="统计信息")
    formatted_patent_results: List[str] = Field(..., description="专利的格式化结果")
    formatted_paper_results: List[str] = Field(..., description="论文的格式化结果")



class ProcessResponse(BaseModel):
    results: Dict[str, str]  # 每个文件的结果以 id 为键
    data: Dict[str, dict]  # 每个文件的结构化数据以 id 为键

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

        doc_type = item.get('类型')

        # 专利处理逻辑
        if doc_type == '专利':
            # 优先使用授权日期，若无则使用申请日期
            date_str = item.get('授权日期')
            if date_str in (None, 'N/A', ''):
                date_str = item.get('申请日期', 'N/A')

            # 如果仍然没有有效日期，则跳过
            if date_str == 'N/A':
                return False

            doc_date = parse_date(date_str)
            return bool(doc_date and start_dt <= doc_date <= end_dt)

        # 论文处理逻辑
        elif doc_type == '论文':
            year = item.get('年份')
            if not year or year == 'N/A':
                return False

            try:
                # 将年份转换为日期（假设为当年1月1日）
                doc_date = datetime(int(year), 1, 1)
                return start_dt <= doc_date <= end_dt
            except (ValueError, TypeError):
                return False

        # 其他类型文档
        return False

    except Exception as e:
        print(f"时效检查错误: {e}")
        return False
@app.post("/api/v1/process_files", response_model=ProcessResponse)
async def process_files(files: List[UploadFile] = File(...)):
    temp_dir = tempfile.mkdtemp()  # 创建临时目录
    results = {}
    structured_data = {}

    try:
        for idx, file in enumerate(files, start=1):
            file_id = f"id{idx}"
            # 第一次尝试提取文本
            text = await pdf_text_reader(file, temp_dir)

            # 检测文档类型
            doc_type = await detect_doc_type(text)
            print("doc_type", doc_type)

            if doc_type == "专利":
                # 提取专利信息
                info = await extract_info(text, "专利")
                info.update({"文件名": file.filename, "类型": "专利"})
                structured_data[file_id] = info
                result = f"文件: {file.filename}\n类型: 专利\n专利号: {info.get('专利号')}\n申请日期: {info.get('申请日期')}\n授权日期: {info.get('授权日期')}\n发明人: {info.get('发明人')}\n受让人: {info.get('受让人')}\n{'=' * 40}"

            elif doc_type == "论文":
                # 提取论文信息
                info = await extract_info(text, "论文")
                info.update({"文件名": file.filename, "类型": "论文"})
                structured_data[file_id] = info
                result = f"""文件: {file.filename}
                        类型: 论文
                        标题: {info.get('标题', 'N/A')}
                        作者: {info.get('作者', 'N/A')}
                        期刊: {info.get('期刊', 'N/A')}
                        年份: {str(info.get('year', 'N/A'))}
                        DOI: {info.get('DOI', 'N/A')}
                        收稿日期: {info.get('received_date', 'N/A')}
                        接受日期: {info.get('accepted_date', 'N/A')}
                        出版日期: {info.get('published_date', 'N/A')}
                        {'=' * 40}"""
            else:
                # 类型未识别，调用 pdf_pic_reader 提取文本
                print(f"未识别的文档类型，尝试通过图片提取文本: {file.filename}")
                text = await pdf_pic_reader(file, temp_dir)  # 使用图片提取文本

                # 重新检测文档类型
                doc_type = await detect_doc_type(text)
                print("重新检测的 doc_type", doc_type)

                if doc_type == "专利":
                    # 提取专利信息
                    info = await extract_info(text, "专利")
                    info.update({"文件名": file.filename, "类型": "专利"})
                    structured_data[file_id] = info
                    result = f"文件: {file.filename}\n类型: 专利\n专利号: {info.get('专利号')}\n申请日期: {info.get('申请日期')}\n授权日期: {info.get('授权日期')}\n发明人: {info.get('发明人')}\n受让人: {info.get('受让人')}\n{'=' * 40}"

                elif doc_type == "论文":
                    # 提取论文信息
                    info = await extract_info(text, "论文")
                    info.update({"文件名": file.filename, "类型": "论文"})
                    structured_data[file_id] = info
                    result = f"""文件: {file.filename}
                            类型: 论文
                            标题: {info.get('标题', 'N/A')}
                            作者: {info.get('作者', 'N/A')}
                            期刊: {info.get('期刊', 'N/A')}
                            年份: {str(info.get('year', 'N/A'))}
                            DOI: {info.get('DOI', 'N/A')}
                            收稿日期: {info.get('received_date', 'N/A')}
                            接受日期: {info.get('accepted_date', 'N/A')}
                            出版日期: {info.get('published_date', 'N/A')}
                            {'=' * 40}"""
                else:
                    # 如果仍未识别，则标记为未识别
                    result = f"文件: {file.filename}\n类型: 未识别\n{'=' * 40}"
                    structured_data[file_id] = {"文件名": file.filename, "类型": "未识别"}

            # 保存结果
            results[file_id] = result
            print(result)

    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir, ignore_errors=True)

    return ProcessResponse(results=results, data=structured_data)

@app.post("/api/v1/check_validity", response_model=ValidityCheckResponse)
async def check_documents_validity(request: ValidityCheckRequest):
    start_dt = parse_date(request.start_date)
    end_dt = parse_date(request.end_date)

    # 验证日期格式
    if not all([start_dt, end_dt]):
        raise HTTPException(status_code=400, detail="日期格式错误")
    if start_dt > end_dt:
        raise HTTPException(status_code=400, detail="起始日期不能晚于结束日期")

    # 初始化结果存储
    valid_patents = []  # 存储有效的专利文档
    valid_papers = []   # 存储有效的论文文档
    formatted_patent_results = []  # 存储专利的格式化结果
    formatted_paper_results = []  # 存储论文的格式化结果
    date_comparisons = []  # 存储日期比较结果
    stats = {
        "patent": {"total": 0, "in_range": 0},
        "paper": {"total": 0, "in_range": 0}
    }

    # 处理专利数据
    for patent_id, patent_doc in request.docs.get("patentData", {}).items():
        date_field = "申请日期"
        date_str = patent_doc.get(date_field, "")
        doc_date = parse_date(date_str) if date_str and date_str not in (None, 'N/A', '') else None

        stats["patent"]["total"] += 1

        if doc_date:
            in_range = start_dt <= doc_date <= end_dt
            date_comparisons.append({
                "filename": patent_doc.get("文件名", ""),
                "doc_type": "专利",
                "date_field": date_field,
                "date_value": date_str,
                "in_range": in_range
            })

            if in_range:
                valid_patents.append(patent_doc)  # 添加到有效专利列表
                stats["patent"]["in_range"] += 1
                formatted = f"""类型: 专利
                                专利号: {patent_doc.get('专利号', 'N/A')}
                                申请日期: {patent_doc.get('申请日期', 'N/A')}
                                授权日期: {patent_doc.get('授权日期', 'N/A')}
                                发明人: {patent_doc.get('发明人', 'N/A')}
                                受让人: {patent_doc.get('受让人', 'N/A')}
                                文件名: {patent_doc.get('文件名', 'N/A')}
                                {'=' * 40}"""
                formatted_patent_results.append(formatted)

    # 处理论文数据
    for paper_id, paper_doc in request.docs.get("paperData", {}).items():
        date_field = "received_date"
        date_str = paper_doc.get(date_field, "")
        doc_date = parse_date(date_str) if date_str and date_str not in (None, 'N/A', '') else None

        stats["paper"]["total"] += 1

        if doc_date:
            in_range = start_dt <= doc_date <= end_dt
            date_comparisons.append({
                "filename": paper_doc.get("文件名", ""),
                "doc_type": "论文",
                "date_field": date_field,
                "date_value": date_str,
                "in_range": in_range
            })

            if in_range:
                valid_papers.append(paper_doc)  # 添加到有效论文列表
                stats["paper"]["in_range"] += 1
                formatted = f"""类型: 论文
                                标题: {paper_doc.get('标题', 'N/A')}
                                作者: {paper_doc.get('作者', 'N/A')}
                                期刊: {paper_doc.get('期刊', 'N/A')}
                                年份: {str(paper_doc.get('year', 'N/A'))}
                                DOI: {paper_doc.get('DOI', 'N/A')}
                                收稿日期: {paper_doc.get('received_date', 'N/A')}
                                接受日期: {paper_doc.get('accepted_date', 'N/A')}
                                出版日期: {paper_doc.get('published_date', 'N/A')}
                                文件名: {paper_doc.get('文件名', 'N/A')}
                                {'=' * 40}"""
                formatted_paper_results.append(formatted)

    return ValidityCheckResponse(
        valid_patents=valid_patents,
        valid_papers=valid_papers,
        total_valid=len(valid_patents) + len(valid_papers),
        time_range=f"{request.start_date} 至 {request.end_date}",
        date_comparisons=date_comparisons,
        comparison_stats=stats,
        formatted_patent_results=formatted_patent_results,
        formatted_paper_results=formatted_paper_results
    )

# 启动服务器
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
