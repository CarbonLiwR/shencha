from ..base.reader import BaseMdReader, BaseCsvReader, BasePdfReader, BaseAudioReader, BaseImageReader, \
    BaseDocxReader, BaseJsonReader, BasePptxReader, BaseXlsxReader, BaseTxtReader
from ..adapter.reader import MdReaderAdapter, DocxReaderAdapter, PdfReaderAdapter, PptxReaderAdapter, XlsxReaderAdapter



class MdReader(BaseMdReader):
    def __init__(self):
        super(MdReader, self).__init__()
        self.md_reader_adapter = MdReaderAdapter()

    def read(self, file_path):
        data = self.md_reader_adapter.read(file_path)
        return data

class PdfReader(BasePdfReader):
    def __init__(self):
        super(PdfReader, self).__init__()
        self.pdf_reader_adapter = PdfReaderAdapter()

    def read(self, file_path):
        data = self.pdf_reader_adapter.read(file_path)
        return data


class DocxReader(BaseDocxReader):
    def __init__(self):
        super(DocxReader, self).__init__()
        self.docx_reader_adapter = DocxReaderAdapter()
    def read(self, file_path):
        data = self.docx_reader_adapter.read(file_path)
        return data

class PptxReader(BasePptxReader):
    def __init__(self):
        super(PptxReader, self).__init__()
        self.pptx_reader_adapter = PptxReaderAdapter()

    def read(self, file_path):
        data = self.pptx_reader_adapter.read(file_path)
        return data

class XlsxReader(BaseXlsxReader):
    def __init__(self):
        super(XlsxReader, self).__init__()
        self.xlsx_reader_adapter = XlsxReaderAdapter()
    def read(self, file_path):
        data = self.xlsx_reader_adapter.read(file_path)
        return data