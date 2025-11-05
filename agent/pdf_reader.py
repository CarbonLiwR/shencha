import os
import aiofiles
import fitz
import pdfplumber
from paddleocr import PaddleOCR
from logging_config import logger
from dotenv import load_dotenv
import os
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from logging_config import logger
# 加载环境变量
load_dotenv()

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# ✅ 指定 Poppler 的 bin 路径
POPPLER_PATH = r"C:\Release-25.07.0-0\poppler-25.07.0\Library\bin"

async def pdf_text_reader(temp_file_path: str) -> str:
    """
    异步处理 PDF 文件，提取文本内容。
    """
    logger.info(f"开始处理PDF文件: {temp_file_path}")

    try:
        with pdfplumber.open(temp_file_path) as pdf:
            all_text = ""
            for page in pdf.pages:
                text0 = page.extract_text()
                if text0:  # ✅ 防止 None 拼接出错
                    all_text += text0 + "\n"
        logger.debug(f"提取的文本内容前200字符: {all_text[:200]}...")
        return all_text
    except Exception as e:
        logger.error(f"PDF解析失败: {str(e)}", exc_info=True)
        return ""


async def extract_text_from_images(image_paths: list) -> str:
    """
    使用 Tesseract OCR 对图片进行识别并提取文本。
    """
    if not image_paths:
        logger.warning("没有提供图片路径")
        return ""

    all_text = []

    for idx, image_path in enumerate(image_paths, 1):
        try:
            logger.info(f"正在处理图片 {idx}/{len(image_paths)}: {image_path}")
            img = Image.open(image_path)

            # 指定中文语言包（需安装 tesseract-ocr-chi-sim）
            text = pytesseract.image_to_string(img, lang="chi_sim+eng")

            logger.info(f"OCR提取内容前100字符: {text[:100]}...")
            all_text.append(text.strip())

        except Exception as e:
            logger.error(f"图片 {image_path} OCR失败: {str(e)}", exc_info=True)

    return "\n".join(all_text).strip()
async def pdf_pic_reader(temp_file_path: str) -> str:
    """
    异步处理 PDF 文件：将每页转为图片后使用 OCR 识别。
    """
    logger.info(f"开始处理PDF文件(图片模式): {temp_file_path}")

    pdf_dir = os.path.dirname(temp_file_path)
    pdf_name = os.path.splitext(os.path.basename(temp_file_path))[0]

    os.makedirs(pdf_dir, exist_ok=True)

    image_paths = []
    try:


        images = convert_from_path(temp_file_path, dpi=300, poppler_path=POPPLER_PATH)

        logger.info(f"PDF总页数: {len(images)}")

        for i, image in enumerate(images, 1):
            image_path = os.path.join(pdf_dir, f"{pdf_name}_page_{i}.png")
            image.save(image_path, "PNG")
            image_paths.append(image_path)
            logger.debug(f"已生成图片: {image_path}")

        if not image_paths:
            logger.error("PDF转图片失败，未生成任何图片。")
            return "PDF转图片失败，无法提取文本内容。"

        # 调用 OCR
        all_text = await extract_text_from_images(image_paths)
        # 清理临时图片
        for img_path in image_paths:
            try:
                os.remove(img_path)
                logger.debug(f"删除临时图片: {img_path}")
            except Exception as e:
                logger.warning(f"删除临时图片失败: {str(e)}")

        return all_text if all_text else "OCR识别未提取到文本内容"

    except Exception as e:
        logger.error(f"PDF转图片处理失败: {str(e)}", exc_info=True)
        return "PDF处理失败，无法提取文本内容"
