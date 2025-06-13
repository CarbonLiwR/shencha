from pathlib import Path
from typing import Any, Type, Optional, Dict
from pydantic import BaseModel, Field



class FunctionSpec(BaseModel):
    function_class: Type                      # 主要逻辑类，如 GraphRetriever
    config_schema_class: Optional[Type] = None  # 对应配置Schema类，如 GraphRetrieverConfigArgs
    default_setting: Optional[Dict[str, Dict[str, Any]]] = None



class DataLoaderConfig(BaseModel):
    pass

class TextExtractorConfig(BaseModel):
    pass

class ImageExtractorConfig(BaseModel):
    save_path: Path

class TableExtractorConfig(BaseModel):
    save_type: str
    save_path: str

class DataChunkerConfig(BaseModel):
    model_name: str
    max_tokens: int

class DoclingServerArgs(BaseModel):
    data_loader: DataLoaderConfig
    text_extractor: TextExtractorConfig
    image_extractor: ImageExtractorConfig
    table_extractor: TableExtractorConfig
    data_chunker: DataChunkerConfig

