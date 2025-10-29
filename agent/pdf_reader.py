import base64
import os

import aiofiles
import fitz
import pdfplumber
from logging_config import logger
from dotenv import load_dotenv
import requests
# 加载环境变量
load_dotenv()
# 获取配置
API_BASE_URL = os.getenv('API_BASE_URL')
API_KEY = os.getenv('API_KEY')
TEXT_MODEL = os.getenv('TEXT_MODEL')
VISION_MODEL = os.getenv('VISION_MODEL')

async def pdf_text_reader(temp_file_path: str) -> str:
    """
    异步处理 PDF 文件，提取文本内容。

    Args:
        temp_file_path (str): 临时文件路径。

    Returns:
        str: 提取的文本内容。
    """
    logger.info(f"开始处理PDF文件: {temp_file_path}")

    # 提取 PDF 文本内容
    try:
        with pdfplumber.open(temp_file_path) as pdf:
            all_text = ""
            for page in pdf.pages:
                text0 = page.extract_text()
                all_text += text0 + "\n"
        logger.debug(f"提取的文本内容前200字符: {all_text[:200]}...")
        return all_text
    except Exception as e:
        logger.error(f"PDF解析失败: {str(e)}", exc_info=True)
        return ""


async def image_to_base64(image_path: str) -> str:
    """
    将图片转换为 Base64 编码。

    Args:
        image_path (str): 图片路径。

    Returns:
        str: Base64 编码的图片。
    """
    try:
        async with aiofiles.open(image_path, "rb") as f:
            image_data = await f.read()
            base64_image = base64.b64encode(image_data).decode("utf-8")
            logger.debug(f"图片转换成功，大小: {len(base64_image)} bytes")
            return base64_image
    except Exception as e:
        logger.error(f"图片转Base64失败: {str(e)}", exc_info=True)
        return ""


async def extract_text_from_images(image_paths: list) -> str:
    """
    使用 GPT 模型对图片进行 OCR 识别并提取文本。

    Args:
        image_paths (list): 图片路径列表。

    Returns:
        str: 提取的文本内容。
    """
    if not image_paths:
        logger.warning("没有提供图片路径")
        return ""

    all_text = []
    url = API_BASE_URL

    for idx, image_path in enumerate(image_paths, 1):
        try:
            logger.info(f"正在处理图片 {idx}/{len(image_paths)}: {image_path}")

            base64_image = await image_to_base64(image_path)
            if not base64_image:
                continue

            payload = {
                "model": VISION_MODEL,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "请完整提取图片的文本信息"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }]
            }

            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }

            logger.debug(f"发送OCR请求，图片大小: {len(base64_image)} bytes")
            response = requests.post(url, json=payload, headers=headers, timeout=(30, 90))
            response.raise_for_status()  # 检查HTTP错误

            response_data = response.json()
            logger.debug(f"OCR响应接收成功，状态码: {response.status_code}")

            page_text = response_data['choices'][0]['message']['content']
            logger.debug(f"提取的文本内容: {page_text[:100]}...")  # 只记录前100字符
            all_text.append(page_text)

        except requests.exceptions.RequestException as e:
            logger.error(f"OCR请求失败: {str(e)}")
        except Exception as e:
            logger.error(f"处理图片 {image_path} 时发生异常: {str(e)}", exc_info=True)

    return "\n".join(all_text) if all_text else ""


async def pdf_pic_reader(temp_file_path: str) -> str:
    """
    异步处理 PDF 文件，转图片后使用 GPT 模型进行 OCR 识别并提取文本内容。

    Args:
        temp_file_path (str): PDF 文件路径。

    Returns:
        str: 提取的文本内容。
    """
    logger.info(f"开始处理PDF文件(图片模式): {temp_file_path}")

    # 获取 PDF 文件所在目录
    pdf_dir = os.path.dirname(temp_file_path)
    pdf_name = os.path.splitext(os.path.basename(temp_file_path))[0]

    # 创建临时目录（如果不存在）
    os.makedirs(pdf_dir, exist_ok=True)
    image_paths = []

    try:
        # 转换PDF为图片
        pdf_document = fitz.open(temp_file_path)
        logger.info(f"PDF总页数: {len(pdf_document)}")

        for page_number in range(len(pdf_document)):
            try:
                page = pdf_document.load_page(page_number)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2倍分辨率

                image_path = os.path.join(pdf_dir, f"{pdf_name}_page_{page_number + 1}.png")
                pix.save(image_path)
                image_paths.append(image_path)
                logger.debug(f"生成图片成功: {image_path}")

            except Exception as e:
                logger.error(f"第 {page_number + 1} 页转图片失败: {str(e)}", exc_info=True)

        if not image_paths:
            logger.error("未能生成任何图片")
            return "PDF转图片失败，无法提取文本内容。"

        logger.info(f"成功生成 {len(image_paths)} 张图片，开始OCR识别")
        all_text = await extract_text_from_images(image_paths)

        # 清理临时图片文件
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
