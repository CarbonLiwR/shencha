import json
import os
import shutil
import tempfile
from datetime import datetime
from typing import List, Dict, Any, Optional

import aiofiles
import aiohttp

import asyncio
from concurrent.futures import ProcessPoolExecutor
executor = ProcessPoolExecutor(max_workers=os.cpu_count() or 4)


# 加载环境变量（如果有）
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from logging_config import logger

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
        logger.info(f"开始下载: {url}")
        logger.info(f"保存路径: {save_path}")

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    # 获取文件大小（可能不可用）
                    file_size = int(response.headers.get('content-length', 0))

                    # 显示下载基本信息
                    logger.info(f"文件大小: {file_size / 1024:.2f} KB" if file_size else "文件大小: 未知")

                    content = b''
                    downloaded = 0
                    async for chunk in response.content.iter_chunked(1024 * 8):  # 8KB chunks
                        content += chunk
                        downloaded += len(chunk)

                        # 显示下载进度（如果有文件大小信息）
                        if file_size > 0:
                            percent = downloaded / file_size * 100
                            logger.debug(f"下载进度: {percent:.1f}% ({downloaded}/{file_size} bytes)")

                    # 保存文件
                    async with aiofiles.open(save_path, "wb") as f:
                        await f.write(content)

                    logger.info(f"下载完成: {url}")
                    return True

                logger.error(f"下载失败: HTTP状态码 {response.status}")
                return False

    except aiohttp.ClientError as e:
        logger.error(f"网络错误: {str(e)}", exc_info=True)
    except IOError as e:
        logger.error(f"文件保存错误: {str(e)}", exc_info=True)
    except Exception as e:
        logger.error(f"未知错误: {str(e)}", exc_info=True)

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

def process_single_file_sync(temp_file_path: str, filename: str) -> tuple[str, dict]:
    """在进程池中运行的同步单文件处理逻辑"""
    import asyncio
    from agent.doc_detecter import detect_doc_type
    from agent.extract_agent import extract_info
    from agent.pdf_reader import pdf_text_reader, pdf_pic_reader

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def inner():
        text = await pdf_text_reader(temp_file_path)
        raw_doc_type = await detect_doc_type(text)
        doc_type = raw_doc_type.split("</think>")[-1].strip()

        if "专利" in doc_type:
            info = await extract_info(text, "专利", filename)
            info.update({"文件名": filename, "类型": "专利"})
            result = f"文件: {filename}\n类型: 专利\n专利号：{info.get('专利号')}\n专利名称: {info.get('专利名称')}\n申请日期: {info.get('申请日期')}\n授权日期: {info.get('授权日期')}\n发明人: {info.get('发明人')}\n受让人: {info.get('受让人')}\n{'=' * 40}"

        elif "论文" in doc_type:
            info = await extract_info(text, "论文", filename)
            info.update({"文件名": filename, "类型": "论文"})
            result = f"""文件: {filename}
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
            info = await extract_info(text, "标准", filename)
            info.update({"文件名": filename, "类型": "标准"})
            result = f"""文件: {filename}
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
            info = await extract_info(text, "软著", filename)
            info.update({"文件名": filename, "类型": "软著"})
            result = f"""文件: {filename}
                                类型: 软著
                                证书号: {info.get('证书号', 'N/A')}
                                软件名称: {info.get('软件名称', 'N/A')}
                                著作权人: {info.get('著作权人', 'N/A')}
                                登记号: {info.get('登记号', 'N/A')}
                                授权时间: {info.get('授权时间', 'N/A')}
                                {'=' * 40}"""

        else:
            # 类型未识别，调用 pdf_pic_reader 提取文本
            logger.info(f"未识别的文档类型，尝试通过图片提取文本: {filename}")
            try:
                text = await pdf_pic_reader(temp_file_path)  # 修改为直接处理临时文件路径
            except Exception as e:
                logger.error(f"PDF 转图片失败: {e}")
                text = None
            logger.debug(f"重新检测的文本内容: {text[:2000] if text else '无文本'}")
            # 重新检测文档类型
            raw_doc_type = await detect_doc_type(text) if text else "其他"
            # 处理doc_type，只保留</think>后的内容
            doc_type = raw_doc_type.split("</think>")[-1].strip()

            logger.debug(f"重新检测的 doc_type: {doc_type}")

            if "专利" in doc_type:
                info = await extract_info(text, "专利", filename)
                info.update({"文件名": filename, "类型": "专利"})
                result = f"文件: {filename}\n类型: 专利\n专利号：{info.get('专利号')}\n专利名称: {info.get('专利名称')}\n申请日期: {info.get('申请日期')}\n授权日期: {info.get('授权日期')}\n发明人: {info.get('发明人')}\n受让人: {info.get('受让人')}\n{'=' * 40}"

            elif "论文" in doc_type:
                info = await extract_info(text, "论文", filename)
                info.update({"文件名": filename, "类型": "论文"})
                result = f"""文件: {filename}
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
                info = await extract_info(text, "标准", filename)
                info.update({"文件名": filename, "类型": "标准"})
                result = f"""文件: {filename}
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
                info = await extract_info(text, "软著", filename)
                info.update({"文件名": filename, "类型": "软著"})
                result = f"""文件: {filename}
                                    类型: 软著
                                    证书号: {info.get('证书号', 'N/A')}
                                    软件名称: {info.get('软件名称', 'N/A')}
                                    著作权人: {info.get('著作权人', 'N/A')}
                                    登记号: {info.get('登记号', 'N/A')}
                                    授权时间: {info.get('授权时间', 'N/A')}
                                    {'=' * 40}"""

            else:
                # 如果仍未识别，则标记为未识别
                result = f"文件: {filename}\n类型: 未识别\n{'=' * 40}"
                info = {"文件名": filename, "类型": "未识别"}

        return result, info

    try:
        result, info = loop.run_until_complete(inner())
    finally:
        asyncio.set_event_loop(None)

    return result, info

@app.post("/api/v1/process_files", response_model=ProcessResponse)
async def process_files(files: List[UploadFile] = File(...)):
    logger.info(f"开始处理文件上传请求，文件数量: {len(files)}")
    temp_dir = tempfile.mkdtemp()
    results = {}
    structured_data = {}

    try:
        tasks = []
        for idx, file in enumerate(files, start=1):
            file_id = f"id{idx}"
            temp_file_path = os.path.join(temp_dir, file.filename)
            logger.info(f"保存文件 {idx}/{len(files)}: {file.filename}")

            # 保存上传文件
            async with aiofiles.open(temp_file_path, "wb") as f:
                content = await file.read()
                await f.write(content)

            # 提交并行任务
            loop = asyncio.get_running_loop()
            task = loop.run_in_executor(executor, process_single_file_sync, temp_file_path, file.filename)
            tasks.append((file_id, task))

        # 并行等待结果
        done_results = await asyncio.gather(*[t for _, t in tasks], return_exceptions=True)
        for (file_id, _), result in zip(tasks, done_results):
            if isinstance(result, Exception):
                results[file_id] = f"文件处理失败: {str(result)}"
                structured_data[file_id] = {"文件名": file_id, "类型": "处理失败"}
            else:
                res, info = result
                results[file_id] = res
                structured_data[file_id] = info



    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return ProcessResponse(results=results, data=structured_data)


# 启动服务器
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
