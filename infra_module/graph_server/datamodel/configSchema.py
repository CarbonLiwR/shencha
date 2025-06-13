from typing import Any, Type, Optional, Dict
from pydantic import BaseModel, Field
from .model_output import EntityExtractedRes
from ..data.prompt.extract import EXTRACT_ENTITIES_FROM_QUERY


class FunctionSpec(BaseModel):
    function_class: Type                      # 主要逻辑类，如 GraphRetriever
    config_schema_class: Optional[Type] = None  # 对应配置Schema类，如 GraphRetrieverConfigArgs
    default_setting: Optional[Dict[str, Dict[str, Any]]] = None



class TextRetrieverConfigArgs(BaseModel):
    embedding_model_path: str


class ExtractModelConfig(BaseModel):
    api_key: str
    base_url: str
    model_name: str
    system_prompt: str
    output_format: Any

class EmbeddingModelConfig(BaseModel):
    api_key: str
    base_url: str
    model_name: str

class SubGraphExtractorConfig(BaseModel):
    graph_level: int | None

class RankerConfig(BaseModel):
    if_reverse: bool

class ContextBuilderConfig(BaseModel):
    max_context_tokens: int
    ranker: RankerConfig

class GraphRetrieverConfig(BaseModel):
    extract_model: ExtractModelConfig
    embedding_model: EmbeddingModelConfig
    sub_graph_extractor: SubGraphExtractorConfig
    ranker: RankerConfig
    context_builder: ContextBuilderConfig


class GraphServerConfigArgs(BaseModel):
    graph_retriever: GraphRetrieverConfig

