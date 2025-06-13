import pandas as pd
from pathlib import Path
from ..base.extractor import Extractor
from docling_core.types.doc import PictureItem


class ImageExtractor(Extractor):
    def __init__(self, save_path: Path):
        super(ImageExtractor, self).__init__()
        self.save_path: Path = save_path


    def extract(self, data):
        path = []
        for result in data:
            doc_filename = result.input.file.stem
            for page_no, page in result.document.pages.items():
                page_image_filename = self.save_path / f"{doc_filename}-{page_no}.png"
                with page_image_filename.open("wb") as fp:
                    page.image.pil_image.save(fp, format="PNG")

                path.append(page_image_filename)

            picture_counter = 0
            for element, _ in result.document.iterate_items():

                if isinstance(element, PictureItem):
                    picture_counter += 1
                    element_image_filename = self.save_path / f"{doc_filename}-picture-{picture_counter}.png"
                    with element_image_filename.open("wb") as fp:
                        element.get_image(result.document).save(fp, "PNG")

                    path.append(element_image_filename)
        return path

class TableExtractor(Extractor):
    def __init__(self,save_path, save_type):
        super(TableExtractor, self).__init__()
        self.save_path = save_path
        self.save_type = save_type

    def extract(self, data):

        rows = []
        path = []

        for result in data:
            doc_filename = result.input.file.stem
            for table_ix, table in enumerate(result.document.tables):
                table_df: pd.DataFrame = table.export_to_dataframe()
                rows.append(table_df)

                if self.save_type == "csv":
                    element_csv_filename = self.save_path / f"{doc_filename}-table-{table_ix + 1}.csv"
                    table_df.to_csv(element_csv_filename, index=False)
                    path.append(element_csv_filename)
                elif self.save_type == "html":
                    element_html_filename = self.save_path / f"{doc_filename}-table-{table_ix + 1}.html"
                    with element_html_filename.open("w") as fp:
                        fp.write(table.export_to_html())
                    path.append(element_html_filename)
                elif self.save_type == "md":
                    element_md_filename = self.save_path / f"{doc_filename}-table-{table_ix + 1}.md"
                    table_df.to_markdown(element_md_filename)
                    path.append(element_md_filename)

        return path

class TextExtractor(Extractor):
    def __init__(self):
        super(TextExtractor, self).__init__()
        pass
    def extract(self, data):
        for result in data:
            return result.document.export_to_markdown()