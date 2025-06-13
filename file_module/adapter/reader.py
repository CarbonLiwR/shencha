from abc import ABC, abstractmethod
from infra_module.docling_server.server import DoclingUserServer



class ReaderAdapter(ABC):
    def __init__(self):
        self.docling_server = DoclingUserServer()

    def read(self, file_path: str):
        data = self.docling_server.load_data(file_path)
        text = self.docling_server.extract_text(data)
        return text

class PdfReaderAdapter(ReaderAdapter):
    def __init__(self):
        super(PdfReaderAdapter, self).__init__()

class DocxReaderAdapter(ReaderAdapter):
    def __init__(self):
        super(DocxReaderAdapter, self).__init__()

class PptxReaderAdapter(ReaderAdapter):
    def __init__(self):
        super(PptxReaderAdapter, self).__init__()

class XlsxReaderAdapter(ReaderAdapter):
    def __init__(self):
        super(XlsxReaderAdapter, self).__init__()

class MdReaderAdapter(ReaderAdapter):
    def __init__(self):
        super(MdReaderAdapter, self).__init__()


