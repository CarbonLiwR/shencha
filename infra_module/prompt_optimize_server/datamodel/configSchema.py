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
    system_prompt: str|None
class AgentConfig(BaseModel):
    model: OpenaiModelConfig

class PromptOptimizeAgentConfig(AgentConfig):
    pass

class PromptAnalyzeAgentConfig(AgentConfig):
    pass

class PromptDetectAgentConfig(AgentConfig):
    pass

class StructuredPromptEditAgentConfig(AgentConfig):
    pass


class PromptOptimizeServerConfigArgs(BaseModel):
    prompt_detect_agent: PromptDetectAgentConfig
    prompt_analyze_agent: PromptAnalyzeAgentConfig
    structured_prompt_edit_agent: StructuredPromptEditAgentConfig
    prompt_optimize_agent: PromptOptimizeAgentConfig

