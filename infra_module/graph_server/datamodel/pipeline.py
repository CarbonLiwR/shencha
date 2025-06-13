from pydantic import BaseModel, Field


class ExtractModelConfig(BaseModel):
    api_key: str = Field(..., description="提取模型API密钥")
    base_url: str = Field(..., description="提取模型接口地址")
    model_name: str = Field(..., description="用于抽取子图的模型名称")
    system_prompt: str = Field(default="请提取实体关系", description="提取系统提示词")
    output_format: str = Field(default="json", description="输出格式（默认json）")

class EmbeddingModelConfig(BaseModel):
    api_key: str = Field(..., description="Embedding模型API密钥")
    base_url: str = Field(..., description="Embedding模型接口地址")
    model_name: str = Field(..., description="Embedding模型名称")

class SubGraphExtractorConfig(BaseModel):
    graph_level: int = Field(default=None, description="子图提取的图层深度")

class RankerConfig(BaseModel):
    if_reverse: bool = Field(default=False, description="是否逆向排序")

class ContextBuilderConfig(BaseModel):
    max_context_tokens: int = Field(default=2048, description="上下文最大Token长度")
    if_reverse: bool = Field(default=False, description="是否逆向排序")

class GraphRetrievePipeline(BaseModel):
    extract_model: ExtractModelConfig
    embedding_model: EmbeddingModelConfig
    sub_graph_extractor: SubGraphExtractorConfig
    ranker: RankerConfig
    context_builder: ContextBuilderConfig
