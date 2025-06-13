from typing import Any, Type, Optional, Dict
from pydantic import BaseModel, Field



class FunctionSpec(BaseModel):
    function_class: Type                      # 主要逻辑类，如 GraphRetriever
    config_schema_class: Optional[Type] = None  # 对应配置Schema类，如 GraphRetrieverConfigArgs
    default_setting: Optional[Dict[str, Dict[str, Any]]] = None

class OpenaiModelConfig(BaseModel):
    base_url: str
    api_key: str
    model_name: str

class OpenaiEmbedModelConfig(OpenaiModelConfig):
    pass


class LocalEmbedModelConfig(BaseModel):
    model_path: str


class RankerConfig(BaseModel):
    if_reverse: bool

class SemanticRetrieverConfig(BaseModel):
    embedding_model: LocalEmbedModelConfig | OpenaiEmbedModelConfig

class RegexChunkerConfig(BaseModel):
    regex_pattern: str

class DataLoaderConfig(BaseModel):
    pass

class LocalDataServerConfigArgs(BaseModel):
    data_retriever: SemanticRetrieverConfig
    data_chunker: RegexChunkerConfig
    data_loader: DataLoaderConfig
    data_ranker: RankerConfig



