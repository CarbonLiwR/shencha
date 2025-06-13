import inspect
from typing import Dict, Any, Type
from .base.server import PromptOptimizeServer
from .component.agent import DetectAgent, AnalysisAgent, StructurePromptEditAgent, OptimizeAgent
from .component.model_call import OpenaiChatModel
from .datamodel.configSchema import FunctionSpec, PromptOptimizeAgentConfig, PromptAnalyzeAgentConfig, PromptDetectAgentConfig, StructuredPromptEditAgentConfig
from .config import args

class PromptOptimizeUserServer(PromptOptimizeServer):
    def __init__(self):
        super(PromptOptimizeUserServer, self).__init__()
        try:
            if self.prompt_detect_agent == None or \
                    self.prompt_analyze_agent == None or \
                    self.sprompt_edit_agent == None or\
                    self.prompt_optimize_agent:
                self.__init_server()
            else:
                pass
        except Exception as e:
            raise RuntimeError(f"[PromptOptimizeServer] Initialization failed: {str(e)}")

    def __init_server(self) -> None:
        self.prompt_detect_agent = DetectAgent(
            model=OpenaiChatModel(
                **vars(args.prompt_detect_agent.model)
            )
        )
        self.prompt_analyze_agent = AnalysisAgent(
            model=OpenaiChatModel(
                **vars(args.prompt_analyze_agent.model)
            )
        )

        self.sprompt_edit_agent = StructurePromptEditAgent(
            model=OpenaiChatModel(
                **vars(args.prompt_analyze_agent.model)
            )
        )

        self.prompt_optimize_agent = OptimizeAgent(
            model=OpenaiChatModel(
                **vars(args.prompt_optimize_agent.model)
            )
        )


class PromptOptimizeDevServer(PromptOptimizeServer):
    def __init__(self):
        super(PromptOptimizeDevServer, self).__init__()
        self.function_mapping = {
            "structured_prompt_edit_agent": FunctionSpec(
                function_class=StructurePromptEditAgent,
                config_schema_class=StructuredPromptEditAgentConfig,
                default_setting=None
            ),
            "prompt_detect_agent": FunctionSpec(
                function_class=DetectAgent,
                config_schema_class=PromptDetectAgentConfig,
                default_setting=None
            ),
            "prompt_analyze_agent": FunctionSpec(
                function_class=AnalysisAgent,
                config_schema_class=PromptAnalyzeAgentConfig,
                default_setting = None
            ),

            "prompt_optimize_agent":FunctionSpec(
                function_class=OptimizeAgent,
                config_schema_class=PromptOptimizeAgentConfig,
                default_setting=None
            )
        }

    def show_config_template(self, name: str) -> Dict[str, Any]:
        if name not in self.function_mapping:
            raise ValueError(f"[DevServer] No config schema found for feature [{name}].")
        spec = self.function_mapping[name]
        if not spec.config_schema_class:
            return {}
        template = spec.config_schema_class.model_json_schema()
        return template

    def __instantiate_with_config(self, cls: Type, config: Dict[str, Any]) -> Any:
        kwargs = {}
        sig = inspect.signature(cls.__init__)
        for name, param in sig.parameters.items():
            if name == 'self':
                continue
            param_type = param.annotation

            if name not in config:
                if param.default is not inspect.Parameter.empty:
                    kwargs[name] = param.default
                else:
                    raise ValueError(f"[DevServer] Missing required config field [{name}] for {cls.__name__}")
                continue

            value = config[name]

            # 如果参数类型是自定义类，且配置是dict，递归实例化
            if hasattr(param_type, '__annotations__') and isinstance(value, dict):
                kwargs[name] = self.__instantiate_with_config(param_type, value)
            else:
                kwargs[name] = value

        return cls(**kwargs)

    def __setting_default_value(self,spec: FunctionSpec, config: Dict[str, Any] | Any = None):
        if spec.default_setting != None:
            for key, value in spec.default_setting.items():
                config[key].update(value)
        return config

    def register_function(self, name: str, config: Dict[str, Any] | Any = None):
        if not hasattr(self, name):
            raise ValueError(f"[DevServer] Attribute [{name}] not defined on DevServer class.")
        if getattr(self, name) is not None:
            raise ValueError(f"[DevServer] Feature [{name}] already registered.")
        if name not in self.function_mapping:
            raise ValueError(f"[DevServer] Feature [{name}] not found in FUNCTION_MAPPING.")
        spec = self.function_mapping[name]
        config = self.__setting_default_value(spec, config)
        instance = self.__instantiate_with_config(spec.function_class, config) if config!={} else spec.function_class()
        setattr(self, name, instance)