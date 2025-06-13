import json
from ..datamodel.data import TextData, TextBlock

class DataLoader():
    def __init__(self):
        pass

    def load_text_data(self, date_path: str):
        with open(date_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        text_blocks = []
        for text_block in data:
            text_blocks.append(TextBlock(content=text_block["content"], embedding=text_block["embedding"]))
        return TextData(text_blocks=text_blocks)