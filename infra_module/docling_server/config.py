import argparse
from .datamodel.configSchema import DoclingServerArgs, DataLoaderConfig, DataChunkerConfig, TableExtractorConfig, ImageExtractorConfig, TextExtractorConfig


def setup_chunker() -> DataChunkerConfig:
    parser = argparse.ArgumentParser(description='graph retriever setting')
    parser.add_argument('--model_name', type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--max_tokens", type=int, default=64)
    return DataChunkerConfig(**vars(parser.parse_args()))


def setup_image_extractor() -> ImageExtractorConfig:
    parser = argparse.ArgumentParser(description='graph retriever setting')
    parser.add_argument('--save_path', type=str, default="save_path")
    return ImageExtractorConfig(**vars(parser.parse_args()))

def setup_table_extractor() -> TableExtractorConfig:
    parser = argparse.ArgumentParser(description='graph retriever setting')
    parser.add_argument('--save_type', type=str, default="md")
    parser.add_argument('--save_path', type=str, default="save_path")
    return TableExtractorConfig(**vars(parser.parse_args()))



def setup_args():
    chunker_args = setup_chunker()
    image_extractor_args = setup_image_extractor()
    table_extractor_args = setup_table_extractor()

    chunker_config = DataChunkerConfig(
        model_name = chunker_args.model_name,
        max_tokens = chunker_args.max_tokens
    )

    image_extractor_config = ImageExtractorConfig(
        save_path = image_extractor_args.save_path
    )

    table_extractor_config = TableExtractorConfig(
        save_type = table_extractor_args.save_type,
        save_path = table_extractor_args.save_path
    )

    return DoclingServerArgs(
        data_loader  = DataLoaderConfig(),
        text_extractor = TextExtractorConfig(),
        image_extractor = image_extractor_config,
        table_extractor = table_extractor_config,
        data_chunker = chunker_config
    )

args = setup_args()