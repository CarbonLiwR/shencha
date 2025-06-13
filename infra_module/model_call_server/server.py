import inspect
from typing import Dict, Any, Type
from .base.server import ModelCallServer
from .component.local import LocalEmbeddingModel
from .component.openai_ import OpenaiEmbeddingModel, OpenaiChatModel
from .datamodel.configSchema import FunctionSpec, OpenaiChatModelConfig, OpenaiEmbedModelConfig, LocalEmbedModelConfig
from .config import args

class ModelCallUserServer(ModelCallServer):
    def __init__(self):
        super(ModelCallUserServer, self).__init__()
        try:
            if self.openai_chat_model == None or \
                    self.openai_embedding_model == None or \
                    self.local_embedding_model==None:
                self.__init_server()
            else:
                pass
        except Exception as e:
            raise RuntimeError(f"[ModelCallServer] Initialization failed: {str(e)}")

    def __init_server(self) -> None:
        self.openai_chat_model = OpenaiChatModel(
            base_url=args.openai_chat_model.base_url,
            api_key=args.openai_chat_model.api_key,
            model_name=args.openai_chat_model.model_name,
            system_prompt=args.openai_chat_model.system_prompt,
            output_format=args.openai_chat_model.output_format
        )
        self.openai_embedding_model = OpenaiEmbeddingModel(
            base_url=args.openai_embedding_model.base_url,
            api_key=args.openai_embedding_model.api_key,
            model_name=args.openai_embedding_model.model_name
        )
        self.local_embedding_model = LocalEmbeddingModel(
            model_path=args.local_embedding_model.model_path
        )

class ModelCallDevServer(ModelCallServer):
    def __init__(self):
        super(ModelCallDevServer, self).__init__()
        self.function_mapping = {
            "openai_chat_model": FunctionSpec(
                function_class=OpenaiChatModel,
                config_schema_class=OpenaiChatModelConfig,
                default_setting = None
            ),
            "openai_embedding_model":FunctionSpec(
                function_class=OpenaiEmbeddingModel,
                config_schema_class=OpenaiEmbedModelConfig,
                default_setting=None
            ),
            "local_embedding_model":FunctionSpec(
                function_class=LocalEmbeddingModel,
                config_schema_class=LocalEmbedModelConfig,
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