from io import BytesIO
from pathlib import Path
from docling_core.types.io import DocumentStream
from ..docling.datamodel.base_models import InputFormat
from ..docling.datamodel.pipeline_options import PdfPipelineOptions
from ..docling.document_converter import DocumentConverter, PdfFormatOption


class DataLoader():
    def __init__(self):
        self.pipeline_options = None
        self.__init_pipline_options()
        self.doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=self.pipeline_options)
            }
        )

    def __init_pipline_options(self):
        if self.pipeline_options == None:
            self.pipeline_options = PdfPipelineOptions()
            self.pipeline_options.images_scale = 2.0
            self.pipeline_options.generate_page_images = True
            self.pipeline_options.generate_picture_images = True
        else:
            pass

    def load_data(self, file_path):
        if isinstance(file_path, str):
            with open(file_path, "rb") as f:
                stream = BytesIO(f.read())
            doc = DocumentStream(name=str(file_path), stream=stream)
            conv_res = self.doc_converter.convert_all(
                [doc],
                raises_on_error=False  # 允许错误情况下继续转换
            )
        else:
            doc_streams = []

            for doc in file_path:
                if isinstance(doc, (str, Path)):
                    with open(doc, "rb") as f:
                        stream = BytesIO(f.read())
                    doc_streams.append(DocumentStream(name=str(doc), stream=stream))
                else:
                    doc_streams.append(doc)

            conv_res = self.doc_converter.convert_all(
                doc_streams,
                raises_on_error=False  # 允许错误情况下继续转换
            )

        return conv_res
