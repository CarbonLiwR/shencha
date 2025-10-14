import base64
import os

import aiofiles
import fitz
import pdfplumber

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
    print(f"处理中: {temp_file_path}")

    # 提取 PDF 文本内容
    try:
        with pdfplumber.open(temp_file_path) as pdf:
            all_text = ""
            for page in pdf.pages:
                text0 = page.extract_text()
                all_text += text0 + "\n"
        return all_text
    except Exception as e:
        print(f"PDF解析失败: {e}")
        return ""


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
    import requests

    url = API_BASE_URL

    for image_path in image_paths:
        base64_image = await image_to_base64(image_path)

        payload = {
            "model": VISION_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "请完整提取图片的文本信息"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"  # 关键修改点2：正确的图片格式
                            }
                        }
                    ]
                },

            ]
        }
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }


        try:
            response = requests.post(url, json=payload, headers=headers)
            response_data = response.json()
            print(response_data)

            page_text = response_data['choices'][0]['message']['content']
            # print(f"提取文本内容: {page_text}")
            all_text += page_text + "\n"
        except Exception as e:
            print(f"图片 OCR 识别失败: {e}")
    return all_text


async def pdf_pic_reader(temp_file_path: str) -> str:
    """
    异步处理 PDF 文件，转图片后使用 GPT 模型进行 OCR 识别并提取文本内容。

    Args:
        temp_file_path (str): PDF 文件路径。

    Returns:
        str: 提取的文本内容。
    """
    print(f"处理中: {temp_file_path}")

    # 获取 PDF 文件所在目录
    pdf_dir = os.path.dirname(temp_file_path)
    pdf_name = os.path.splitext(os.path.basename(temp_file_path))[0]  # 获取 PDF 文件名（无扩展名）

    # 转换 PDF 为图片
    image_paths = []
    try:
        pdf_document = fitz.open(temp_file_path)

        for page_number in range(len(pdf_document)):
            page = pdf_document.load_page(page_number)
            zoom = 2  # 缩放比例，1表示原始分辨率，0.5表示降低分辨率，2表示提高分辨率
            matrix = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=matrix, annots=False)  # 应用缩放矩阵

            # 图片保存路径：与 PDF 文件同目录，文件名为 PDF 文件名加页码
            image_path = os.path.join(pdf_dir, f"{pdf_name}_page_{page_number + 1}.png")
            # print(f"保存图片路径: {image_path}")
            pix.save(image_path)
            image_paths.append(image_path)
            print(f"生成图片: {image_path}")
        if not image_paths:
            print("PDF 转图片失败，无法提取文本内容。")
            return "PDF 转图片失败，无法提取文本内容。"
    except Exception as e:
        print(f"PDF 转图片失败: {e}")
        return "PDF 转图片失败，无法提取文本内容。"

    # 使用 GPT 模型对图片进行 OCR 识别
    try:
        all_text = await extract_text_from_images(image_paths)
        return all_text
    except Exception as e:
        print(f"OCR 识别失败: {e}")
        return "OCR 识别失败，无法提取文本内容。"
