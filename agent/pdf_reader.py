import asyncio
import base64
import os
import aiofiles
import fitz
import pdfplumber
from fastapi import UploadFile, File
from openai import AsyncOpenAI
from tempfile import TemporaryDirectory


async def pdf_text_reader(file: UploadFile, temp_dir: str) -> str:
    """
    异步处理 PDF 文件，保存到临时目录并提取文本内容。

    Args:
        file (UploadFile): FastAPI 上传的文件对象。
        temp_dir (str): 临时目录路径。

    Returns:
        str: 提取的文本内容。
    """
    file_path = os.path.join(temp_dir, file.filename)
    print(f"处理中: {file.filename}")

    # 异步保存文件
    async with aiofiles.open(file_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    # 提取 PDF 文本内容
    try:
        with pdfplumber.open(file_path) as pdf:
            all_text = ""
            for page in pdf.pages:
                text0 = page.extract_text()
                all_text += text0 + "\n"
        return all_text
    except Exception as e:
        print(f"PDF解析失败: {e}")
        return ""

# 初始化大模型客户端
client = AsyncOpenAI(
    base_url="https://api.rcouyi.com/v1",
    api_key="sk-pAauG9ss64pW9FVA703F1453b334eFb95B7447b9083BaBd"
)
async def pdf_to_images(file_path: str, temp_dir: str) -> list:
    """
    将 PDF 转换为图片并保存到临时目录。

    Args:
        file_path (str): PDF 文件路径。
        temp_dir (str): 临时目录路径。

    Returns:
        list: 保存的图片路径列表。
    """
    image_paths = []
    try:
        pdf_document = fitz.open(file_path)
        for page_number in range(len(pdf_document)):
            page = pdf_document.load_page(page_number)
            pix = page.get_pixmap()
            image_path = os.path.join(temp_dir, f"page_{page_number + 1}.png")
            pix.save(image_path)
            image_paths.append(image_path)
        return image_paths
    except Exception as e:
        print(f"PDF 转图片失败: {e}")
        return []


async def image_to_base64(image_path: str) -> str:
    """
    将图片转换为 Base64 编码。

    Args:
        image_path (str): 图片路径。

    Returns:
        str: Base64 编码的图片。
    """
    async with aiofiles.open(image_path, "rb") as f:
        image_data = await f.read()
        base64_image = base64.b64encode(image_data).decode("utf-8")
        return base64_image


async def extract_text_from_images(image_paths: list) -> str:
    """
    使用 GPT 模型对图片进行 OCR 识别并提取文本。

    Args:
        image_paths (list): 图片路径列表。

    Returns:
        str: 提取的文本内容。
    """
    all_text = ""
    for image_path in image_paths:
        base64_image = await image_to_base64(image_path)
        prompt = "请从以下图片中提取文本内容："
        try:
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "你是一个专业的 OCR 文本提取助手"},
                    {"role": "user", "content": prompt},
                    {"role": "user", "content": f"data:image/png;base64,{base64_image}"}
                ],
                temperature=0
            )
            page_text = response.choices[0].message.content
            all_text += page_text + "\n"
        except Exception as e:
            print(f"图片 OCR 识别失败: {e}")
    return all_text


async def pdf_pic_reader(file: UploadFile, temp_dir: str) -> str:
    """
    异步处理 PDF 文件，转图片后使用 GPT 模型进行 OCR 识别并提取文本内容。

    Args:
        file (UploadFile): FastAPI 上传的文件对象。
        temp_dir (str): 临时目录路径。

    Returns:
        str: 提取的文本内容。
    """
    file_path = os.path.join(temp_dir, file.filename)
    print(f"处理中: {file.filename}")

    # 异步保存文件
    async with aiofiles.open(file_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    # 转换 PDF 为图片
    image_paths = await pdf_to_images(file_path, temp_dir)
    if not image_paths:
        return "PDF 转图片失败，无法提取文本内容。"

    # 使用 GPT 模型对图片进行 OCR 识别
    all_text = await extract_text_from_images(image_paths)
    return all_text