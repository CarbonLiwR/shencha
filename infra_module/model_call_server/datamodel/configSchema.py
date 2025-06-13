from typing import Any, Type, Optional, Dict
from pydantic import BaseModel, Field



class FunctionSpec(BaseModel):
    function_class: Type
    config_schema_class: Optional[Type] = None
    default_setting: Optional[Dict[str, Dict[str, Any]]] = None

class OpenaiModelConfig(BaseModel):
    base_url: str
    api_key: str
    model_name: str

class OpenaiChatModelConfig(OpenaiModelConfig):
    system_prompt: str
    output_format: Any

class OpenaiEmbedModelConfig(OpenaiModelConfig):
    pass


class LocalEmbedModelConfig(BaseModel):
    model_path: str

class ModelCallServerConfigArgs(BaseModel):
    openai_chat_model: OpenaiChatModelConfig
    openai_embedding_model: OpenaiEmbedModelConfig
    local_embedding_model: LocalEmbedModelConfig
