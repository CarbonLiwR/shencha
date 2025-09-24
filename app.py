import json
import os
import shutil
import tempfile
from datetime import datetime
from typing import List, Dict, Any, Optional

import aiofiles
import aiohttp
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
    valid_patents: List[Dict[str, Any]] = Field(default_factory=list, description="有效的专利文档")
    valid_papers: List[Dict[str, Any]] = Field(default_factory=list, description="有效的论文文档")
    valid_standards: List[Dict[str, Any]] = Field(default_factory=list, description="有效的标准文档")
    valid_copyrights: List[Dict[str, Any]] = Field(default_factory=list, description="有效的软件著作权文档")
    total_valid: int = Field(0, description="有效文档总数")
    time_range: str = Field("", description="时间范围")
    date_comparisons: List[Dict[str, Any]] = Field(default_factory=list, description="日期比较结果")
    comparison_stats: Dict[str, Dict[str, int]] = Field(default_factory=dict, description="统计信息")
    formatted_patent_results: List[str] = Field(default_factory=list, description="专利的格式化结果")
    formatted_paper_results: List[str] = Field(default_factory=list, description="论文的格式化结果")
    formatted_standard_results: List[str] = Field(default_factory=list, description="标准的格式化结果")
    formatted_copyright_results: List[str] = Field(default_factory=list, description="软著的格式化结果")

class ProcessResponse(BaseModel):
    results: Dict[str, str]  # 每个文件的结果以 id 为键
    data: Dict[str, dict]  # 每个文件的结构化数据以 id 为键


async def download_from_url(url: str, save_path: str) -> bool:
    """下载文件并显示进度信息"""
    try:
        print(f"⏳ 开始下载: {url}")
        print(f"📁 保存路径: {save_path}")

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    # 获取文件大小（可能不可用）
                    file_size = int(response.headers.get('content-length', 0))

                    # 显示下载基本信息
                    print(f"📦 文件大小: {file_size / 1024:.2f} KB" if file_size else "📦 文件大小: 未知")

                    content = b''
                    downloaded = 0
                    async for chunk in response.content.iter_chunked(1024 * 8):  # 8KB chunks
                        content += chunk
                        downloaded += len(chunk)

                        # 显示下载进度（如果有文件大小信息）
                        if file_size > 0:
                            percent = downloaded / file_size * 100
                            print(f"⬇️ 下载进度: {percent:.1f}% ({downloaded}/{file_size} bytes)", end='\r')

                    # 保存文件
                    async with aiofiles.open(save_path, "wb") as f:
                        await f.write(content)

                    print(f"\n✅ 下载完成: {url}")
                    return True

                print(f"❌ 下载失败: HTTP状态码 {response.status}")
                return False

    except aiohttp.ClientError as e:
        print(f"❌ 网络错误: {str(e)}")
    except IOError as e:
        print(f"❌ 文件保存错误: {str(e)}")
    except Exception as e:
        print(f"❌ 未知错误: {str(e)}")

    return False


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
        if "专利" in doc_type:
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
        elif '论文' in doc_type :
            year = item.get('年份')
            if not year or year == 'N/A':
                return False

            try:
                # 将年份转换为日期（假设为当年1月1日）
                doc_date = datetime(int(year), 1, 1)
                return start_dt <= doc_date <= end_dt
            except (ValueError, TypeError):
                return False

        # 标准处理逻辑
        elif '标准' in doc_type:
            # 优先使用发布时间，若无则使用实施时间
            date_str = item.get('发布时间')
            if date_str in (None, 'N/A', ''):
                date_str = item.get('实施时间', 'N/A')

            # 如果仍然没有有效日期，则跳过
            if date_str == 'N/A':
                return False

            doc_date = parse_date(date_str)
            return bool(doc_date and start_dt <= doc_date <= end_dt)

        # 软著处理逻辑
        elif '软著' in doc_type:
            date_str = item.get('授权时间')
            if date_str in (None, 'N/A', ''):
                return False

            doc_date = parse_date(date_str)
            return bool(doc_date and start_dt <= doc_date <= end_dt)

        # 其他类型文档
        return False

    except Exception as e:
        print(f"时效检查错误: {e}")
        return False


@app.post("/api/v1/process_files", response_model=ProcessResponse)
async def process_files(files: List[UploadFile] = File(...)):
    temp_dir = tempfile.mkdtemp()
    results = {}
    structured_data = {}

    try:
        for idx, file in enumerate(files, start=1):
            file_id = f"id{idx}"
            temp_file_path = os.path.join(temp_dir, file.filename)
            url = None
            is_url = False

            # 改进的URL文件判断逻辑
            try:
                # 方法1：检查type属性
                if getattr(file, 'type', None) == 'url':
                    is_url = True
                    url = getattr(file, 'url', '')

                # 方法2：检查文件内容是否为JSON格式的URL信息
                if not is_url:
                    content = await file.read()
                    try:
                        url_info = json.loads(content)
                        if isinstance(url_info, dict) and 'url' in url_info:
                            is_url = True
                            url = url_info['url']
                    except json.JSONDecodeError:
                        pass
                    await file.seek(0)  # 重置文件指针
            except Exception as e:
                pass

            # 文件内容获取逻辑
            try:
                if is_url and url:
                    print(f"下载URL文件: {url}")
                    if not await download_from_url(url, temp_file_path):
                        # 修改为返回结构化错误信息
                        raise HTTPException(
                            status_code=400,
                            detail={
                                "error": "URL_DOWNLOAD_FAILED",
                                "message": f"URL文件下载失败: {url}",
                                "filename": file.filename,
                                "url": url,
                                "file_id": file_id
                            }
                        )

                else:
                    # 普通文件处理
                    async with aiofiles.open(temp_file_path, "wb") as f:
                        await file.seek(0)
                        content = await file.read()
                        await f.write(content)
            except HTTPException as e:
                # 直接重新抛出HTTPException
                raise e
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "FILE_PROCESSING_ERROR",
                        "message": str(e),
                        "filename": file.filename,
                        "file_id": file_id,
                        **({"url": url} if is_url else {})
                    }
                )

            # 后续处理保持不变...
            text = await pdf_text_reader(temp_file_path)
            raw_doc_type = await detect_doc_type(text)
            # 处理doc_type，只保留</think>后的内容
            print("大模型",raw_doc_type)
            doc_type = raw_doc_type.split("</think>")[-1].strip()
            print(f"检测到的文档类型111: {doc_type}")
            print("111", text[:12000])
            print("111", text[-12000:])
            

            if "专利" in doc_type:
                # 提取专利信息

                info = await extract_info(text, "专利", file.filename)
                info.update({"文件名": file.filename, "类型": "专利"})
                structured_data[file_id] = info
                result = f"文件: {file.filename}\n类型: 专利\n专利号：{info.get('专利号')}\n专利名称: {info.get('专利名称')}\n申请日期: {info.get('申请日期')}\n授权日期: {info.get('授权日期')}\n发明人: {info.get('发明人')}\n受让人: {info.get('受让人')}\n{'=' * 40}"

            elif "论文" in doc_type:
                # 提取论文信息
                info = await extract_info(text, "论文", file.filename)
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
                        项目编号: {info.get('project_number', 'N/A')} 
                        单位: {info.get('institution', 'N/A')}    
                        {'=' * 40}"""

            elif "标准" in doc_type:
                # 提取标准信息
                info = await extract_info(text, "标准", file.filename)
                info.update({"文件名": file.filename, "类型": "标准"})
                structured_data[file_id] = info
                result = f"""文件: {file.filename}
                        类型: 标准
                        标准名称: {info.get('标准名称', 'N/A')}
                        标准形式: {info.get('标准形式', 'N/A')}
                        标准编号: {info.get('标准编号', 'N/A')}
                        起草单位: {info.get('起草单位', 'N/A')}
                        起草人: {info.get('起草人', 'N/A')}
                        发布单位: {info.get('发布单位', 'N/A')}
                        发布时间: {info.get('发布时间', 'N/A')}
                        实施时间: {info.get('实施时间', 'N/A')}
                        {'=' * 40}"""

            elif "软著" in doc_type:
                # 提取软著信息
                info = await extract_info(text, "软著", file.filename)
                info.update({"文件名": file.filename, "类型": "软著"})
                structured_data[file_id] = info
                result = f"""文件: {file.filename}
                        类型: 软著
                        证书号: {info.get('证书号', 'N/A')}
                        软件名称: {info.get('软件名称', 'N/A')}
                        著作权人: {info.get('著作权人', 'N/A')}
                        登记号: {info.get('登记号', 'N/A')}
                        授权时间: {info.get('授权时间', 'N/A')}
                        {'=' * 40}"""
            else:
                # 类型未识别，调用 pdf_pic_reader 提取文本
                print(f"未识别的文档类型，尝试通过图片提取文本: {file.filename}")
                try:
                    text = await pdf_pic_reader(temp_file_path)  # 修改为直接处理临时文件路径
                except Exception as e:
                    print(f"PDF 转图片失败: {e}")
                    text = None
                print(f"重新检测的文本内容: {text[:12000] if text else '无文本'}")
                # 重新检测文档类型
                raw_doc_type = await detect_doc_type(text)if text else "其他"
                # 处理doc_type，只保留</think>后的内容
                doc_type = raw_doc_type.split("</think>")[-1].strip()

                print("重新检测的 doc_type", doc_type)

                if "专利" in doc_type:
                    # 提取专利信息
                    info = await extract_info(text, "专利", file.filename)
                    info.update({"文件名": file.filename, "类型": "专利"})
                    structured_data[file_id] = info
                    result = f"文件: {file.filename}\n类型: 专利\n专利号: {info.get('专利号')}\n专利名称: {info.get('专利名称')}\n申请日期: {info.get('申请日期')}\n授权日期: {info.get('授权日期')}\n发明人: {info.get('发明人')}\n受让人: {info.get('受让人')}\n{'=' * 40}"

                elif "论文" in doc_type:
                    # 提取论文信息
                    info = await extract_info(text, "论文", file.filename)
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
                                            项目编号: {info.get('project_number', 'N/A')} 
                                            单位: {info.get('institution', 'N/A')} 
                                            {'=' * 40}"""

                elif "标准" in doc_type:
                    # 提取标准信息
                    info = await extract_info(text, "标准", file.filename)
                    info.update({"文件名": file.filename, "类型": "标准"})
                    structured_data[file_id] = info
                    result = f"""文件: {file.filename}
                            类型: 标准
                            标准名称: {info.get('标准名称', 'N/A')}
                            标准形式: {info.get('标准形式', 'N/A')}
                            标准编号: {info.get('标准编号', 'N/A')}
                            起草单位: {info.get('起草单位', 'N/A')}
                            起草人: {info.get('起草人', 'N/A')}
                            发布单位: {info.get('发布单位', 'N/A')}
                            发布时间: {info.get('发布时间', 'N/A')}
                            实施时间: {info.get('实施时间', 'N/A')}
                            {'=' * 40}"""

                elif "软著" in doc_type :
                    # 提取标准信息
                    info = await extract_info(text, "软著", file.filename)
                    info.update({"文件名": file.filename, "类型": "软著"})
                    structured_data[file_id] = info
                    result = f"""文件: {file.filename}
                            类型: 软著
                            证书号: {info.get('证书号', 'N/A')}
                            软件名称: {info.get('软件名称', 'N/A')}
                            著作权人: {info.get('著作权人', 'N/A')}
                            登记号: {info.get('登记号', 'N/A')}
                            授权时间: {info.get('授权时间', 'N/A')}
                            {'=' * 40}"""

                else:
                    # 如果仍未识别，则标记为未识别
                    result = f"文件: {file.filename}\n类型: 未识别\n{'=' * 40}"
                    structured_data[file_id] = {"文件名": file.filename, "类型": "未识别"}
            print("result1",result)
            # 保存结果
            results[file_id] = result
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
    valid_papers = []  # 存储有效的论文文档
    valid_standards = [] # 存储有效的标准文档
    valid_copyrights = []  # 存储有效的软著文档
    formatted_patent_results = []  # 存储专利的格式化结果
    formatted_paper_results = []  # 存储论文的格式化结果
    formatted_standard_results = []  # 存储标准的格式化结果
    formatted_copyright_results = []  # 存储软著的格式化结果
    date_comparisons = []  # 存储日期比较结果
    stats = {
        "patent": {"total": 0, "in_range": 0},
        "paper": {"total": 0, "in_range": 0},
        "standard": {"total": 0, "in_range": 0},
        "copyright": {"total": 0, "in_range": 0}
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
                                专利名称: {patent_doc.get('专利名称', 'N/A')}
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
                                项目编号: {paper_doc.get('project_number', 'N/A')}  # 新增行
                                单位: {paper_doc.get('institution', 'N/A')}  # 新增行
                                文件名: {paper_doc.get('文件名', 'N/A')}
                                {'=' * 40}"""
                formatted_paper_results.append(formatted)

    for standard_id, standard_doc in request.docs.get("standardData", {}).items():
        date_field = "发布时间"
        date_str = standard_doc.get(date_field, "")
        doc_date = parse_date(date_str) if date_str and date_str not in (None, 'N/A', '') else None

        stats["standard"]["total"] += 1

        if doc_date:
            in_range = start_dt <= doc_date <= end_dt
            date_comparisons.append({
                "filename": standard_doc.get("文件名", ""),
                "doc_type": "标准",
                "date_field": date_field,
                "date_value": date_str,
                "in_range": in_range
            })

            if in_range:
                valid_standards.append(standard_doc)
                stats["standard"]["in_range"] += 1
                formatted = f"""类型: 标准
                                标准名称: {standard_doc.get('标准名称', 'N/A')}
                                标准形式: {standard_doc.get('标准形式', 'N/A')}
                                标准编号: {standard_doc.get('标准编号', 'N/A')}
                                发布单位: {standard_doc.get('发布单位', 'N/A')}
                                发布时间: {standard_doc.get('发布时间', 'N/A')}
                                实施时间: {standard_doc.get('实施时间', 'N/A')}
                                文件名: {standard_doc.get('文件名', 'N/A')}
                                {'=' * 40}"""
                formatted_standard_results.append(formatted)

    # 处理软著数据
    for copyright_id, copyright_doc in request.docs.get("copyrightData", {}).items():
        date_field = "授权时间"
        date_str = copyright_doc.get(date_field, "")
        doc_date = parse_date(date_str) if date_str and date_str not in (None, 'N/A', '') else None

        stats["copyright"] = stats.get("copyright", {"total": 0, "in_range": 0})
        stats["copyright"]["total"] += 1

        if doc_date:
            in_range = start_dt <= doc_date <= end_dt
            date_comparisons.append({
                "filename": copyright_doc.get("文件名", ""),
                "doc_type": "软著",
                "date_field": date_field,
                "date_value": date_str,
                "in_range": in_range
            })

            if in_range:
                valid_copyrights.append(copyright_doc)
                stats["copyright"]["in_range"] += 1
                formatted = f"""类型: 软著
                                证书号: {copyright_doc.get('证书号', 'N/A')}
                                软件名称: {copyright_doc.get('软件名称', 'N/A')}
                                著作权人: {copyright_doc.get('著作权人', 'N/A')}
                                登记号: {copyright_doc.get('登记号', 'N/A')}
                                授权时间: {copyright_doc.get('授权时间', 'N/A')}
                                文件名: {copyright_doc.get('文件名', 'N/A')}
                                {'=' * 40}"""
                formatted_copyright_results.append(formatted)

    return ValidityCheckResponse(
        valid_patents=valid_patents,
        valid_papers=valid_papers,
        valid_standards=valid_standards,  # 确保包含此字段
        valid_copyrights=valid_copyrights,
        total_valid=len(valid_patents) + len(valid_papers) + len(valid_standards)+ len(valid_copyrights),  # 更新总数计算
        time_range=f"{request.start_date} 至 {request.end_date}",
        date_comparisons=date_comparisons,
        comparison_stats=stats,
        formatted_patent_results=formatted_patent_results,
        formatted_paper_results=formatted_paper_results,
        formatted_standard_results=formatted_standard_results,
        formatted_copyright_results=formatted_copyright_results  # 添加软著格式化结果

    )


# 启动服务器
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
