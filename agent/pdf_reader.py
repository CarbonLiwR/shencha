import base64
import os
import aiofiles
import fitz
import pdfplumber
import asyncio
import aiohttp
from logging_config import logger
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()
API_BASE_URL = os.getenv("API_BASE_URL")
API_KEYS = os.getenv("API_KEYS", os.getenv("API_KEY", "")).split(",")  # 支持单key或多key
TEXT_MODEL = os.getenv("TEXT_MODEL")
VISION_MODEL = os.getenv("VISION_MODEL")

MAX_CONCURRENCY = 3  # 最大并行请求数
MAX_RETRIES = 2       # 每张图片失败重试次数


async def pdf_text_reader(temp_file_path: str) -> str:
    logger.info(f"开始处理PDF文件: {temp_file_path}")
    try:
        with pdfplumber.open(temp_file_path) as pdf:
            all_text = ""
            for page in pdf.pages:
                text0 = page.extract_text()
                if text0:
                    all_text += text0 + "\n"
        logger.debug(f"提取的文本内容前200字符: {all_text[:200]}...")
        return all_text
    except Exception as e:
        logger.error(f"PDF解析失败: {str(e)}", exc_info=True)
        return ""


async def image_to_base64(image_path: str) -> str:
    try:
        async with aiofiles.open(image_path, "rb") as f:
            image_data = await f.read()
            base64_image = base64.b64encode(image_data).decode("utf-8")
            logger.debug(f"图片转换成功，大小: {len(base64_image)} bytes")
            return base64_image
    except Exception as e:
        logger.error(f"图片转Base64失败: {str(e)}", exc_info=True)
        return ""


async def _ocr_single_image(session, image_path: str, api_key: str, idx: int) -> tuple[int, str]:
    """单张图片异步OCR任务，返回(索引,文本)"""
    for attempt in range(MAX_RETRIES + 1):
        try:
            base64_image = await image_to_base64(image_path)
            if not base64_image:
                return idx, ""

            payload = {
                "model": VISION_MODEL,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "请完整提取图片的文本信息"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]
                }]
            }
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

            async with session.post(API_BASE_URL, json=payload, headers=headers,
                                    timeout=aiohttp.ClientTimeout(total=400)) as resp:
                resp.raise_for_status()
                data = await resp.json()
                text = data["choices"][0]["message"]["content"]
                logger.info(f"OCR成功: {os.path.basename(image_path)}")
                return idx, text

        except Exception as e:
            logger.warning(f"OCR第{attempt + 1}次失败: {image_path} | 错误: {e}")
            await asyncio.sleep(2 ** attempt + 0.5)# 递增等待
    return idx, ""


async def extract_text_from_images(image_paths: list) -> str:
    """使用 GPT 模型对图片进行并行 OCR"""
    if not image_paths:
        logger.warning("没有提供图片路径")
        return ""

    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    async with aiohttp.ClientSession() as session:
        async def bound_task(idx, path):
            async with sem:
                api_key = API_KEYS[idx % len(API_KEYS)].strip()
                return await _ocr_single_image(session, path, api_key, idx)

        tasks = [bound_task(i, path) for i, path in enumerate(image_paths)]
        results = await asyncio.gather(*tasks)

    # 保持原始顺序
    results.sort(key=lambda x: x[0])
    all_text = "\n".join(t for _, t in results if t)
    return all_text


async def pdf_pic_reader(temp_file_path: str) -> str:
    """PDF 转图片后使用 GPT 模型并行 OCR"""
    logger.info(f"开始处理PDF文件(图片模式): {temp_file_path}")

    pdf_dir = os.path.dirname(temp_file_path)
    pdf_name = os.path.splitext(os.path.basename(temp_file_path))[0]
    os.makedirs(pdf_dir, exist_ok=True)
    image_paths = []

    try:
        pdf_document = fitz.open(temp_file_path)
        logger.info(f"PDF总页数: {len(pdf_document)}")

        for page_number in range(len(pdf_document)):
            try:
                page = pdf_document.load_page(page_number)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                image_path = os.path.join(pdf_dir, f"{pdf_name}_page_{page_number + 1}.png")
                pix.save(image_path)
                image_paths.append(image_path)
                logger.debug(f"生成图片成功: {image_path}")
            except Exception as e:
                logger.error(f"第 {page_number + 1} 页转图片失败: {str(e)}", exc_info=True)

        if not image_paths:
            logger.error("未能生成任何图片")
            return "PDF转图片失败，无法提取文本内容。"

        logger.info(f"成功生成 {len(image_paths)} 张图片，开始并行OCR识别")
        all_text = await extract_text_from_images(image_paths)

        # 清理图片
        for img_path in image_paths:
            try:
                os.remove(img_path)
                logger.debug(f"已删除临时图片: {img_path}")
            except Exception as e:
                logger.warning(f"删除临时图片失败: {str(e)}")

        return all_text if all_text else "OCR识别未提取到文本内容"

    except Exception as e:
        logger.error(f"PDF图片处理失败: {str(e)}", exc_info=True)
        return "PDF处理失败，无法提取文本内容"
