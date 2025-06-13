from abc import ABC, abstractmethod
from ..component.data_loader import DataLoader
from ..component.chunker import Chunker
from ..component.extractor import TableExtractor, TextExtractor, ImageExtractor


class DoclingServer(ABC):
    def __init__(self):
        self.data_loader: DataLoader| None = None
        self.text_extractor: TextExtractor | None = None
        self.image_extractor: ImageExtractor | None = None
        self.table_extractor: TableExtractor | None = None
        self.data_chunker: Chunker | None = None


    def load_data(self, file_path):
        data = self.data_loader.load_data(file_path)
        return data

    def extract_text(self, data):
        text = self.text_extractor.extract(data)
        return text

    def extract_image(self, data):
        image_path = self.image_extractor.extract(data)
        return image_path

    def extract_table(self, data):
        table_path = self.table_extractor.extract(data)
        return table_path

    def chunk_data(self, text):
        chunked_data = self.data_chunker.chunk(text)
        return chunked_data