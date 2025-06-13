from .magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from .magic_pdf.data.dataset import PymuDocDataset
from .magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from .magic_pdf.config.enums import SupportedPdfParseMethod
# mineru_server.py
image_dir = ""
image_writer = FileBasedDataWriter("")

class TextExtractor():
    def __init__(self):
        self.data_reader = FileBasedDataReader("")
        # self.image_dir = ""
        # os.makedirs(self.image_dir, exist_ok=True)

        # self.image_writer = FileBasedDataWriter("")

    def __read_pdf_bytes_by_data_reader(self, file_path):
        pdf_bytes = self.data_reader.read(file_path)
        return pdf_bytes

    # def __analyze_doc(self, pdf_bytes):
    #     pass


    # def __handle_analysis_result(self, analysis_result, image_writer):
    #     pass
    
    def extract_text(self, file_path):
        pdf_bytes = self.__read_pdf_bytes_by_data_reader(file_path)
        ds = PymuDocDataset(pdf_bytes)
        if ds.classify() == SupportedPdfParseMethod.OCR:
            infer_result = ds.apply(doc_analyze, ocr=True)
            ## pipeline
            pipe_result = infer_result.pipe_ocr_mode(image_writer)

        else:
            infer_result = ds.apply(doc_analyze, ocr=False)

            ## pipeline
            pipe_result = infer_result.pipe_txt_mode(image_writer)
        md_content = pipe_result.get_markdown(image_dir)
        return md_content