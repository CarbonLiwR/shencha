import types
import inspect
from typing import Any, Dict, Type, get_origin, get_args, Union
from .base.server import LocalDataServer
from .component.retriever import TextDataSemanticRetriever
from .component.loader import DataLoader
from .component.ranker import FieldRanker
from .component.chunker import RegexChunker
from .component.modelcall import LocalEmbeddingModel
from .datamodel.configSchema import FunctionSpec
from .datamodel.configSchema import SemanticRetrieverConfig, RegexChunkerConfig, RankerConfig
from .config import args

class LocalDataUserServer(LocalDataServer):
    def __init__(self):
        super(LocalDataUserServer, self).__init__()
        try:
            if self.data_semantic_retriever == None or \
                    self.regex_chunker == None or \
                    self.data_ranker==None or \
                    self.data_loader==None:
                self.__init_server()
            else:
                pass
        except Exception as e:
            raise RuntimeError(f"[LocalDataServer] Initialization failed: {str(e)}")

    def __init_server(self) -> None:
        self.data_semantic_retriever = TextDataSemanticRetriever(
            embedding_model= LocalEmbeddingModel(
                    model_path = args.data_retriever.embedding_model.model_path
                ),
        )
        self.regex_chunker = RegexChunker(pattern=args.data_chunker.regex_pattern)
        self.data_ranker = FieldRanker(if_reverse=args.data_ranker.if_reverse)
        self.data_loader = DataLoader()

class LocalDataDevServer(LocalDataServer):
    def __init__(self):
        super(LocalDataDevServer, self).__init__()
        self.function_mapping = {
            "data_loader":FunctionSpec(
                function_class=DataLoader,
                config_schema_class=None,
                default_setting = None
            ),
            "data_semantic_retriever": FunctionSpec(
                function_class=TextDataSemanticRetriever,
                config_schema_class=SemanticRetrieverConfig,
                default_setting = None
            ),
            "field_ranker":FunctionSpec(
                function_class=FieldRanker,
                config_schema_class=RankerConfig,
                default_setting=None
            ),
            "regex_chunker":FunctionSpec(
                function_class=RegexChunker,
                config_schema_class=RegexChunkerConfig,
                default_setting=None
            )
        }

    def show_config_template(self, name: str) -> Dict[str, Any]:
        if name not in self.function_mapping:
            raise ValueError(f"[DevServer] No config schema found for feature [{name}].")
        spec = self.function_mapping[name]
        if not spec.config_schema_class:
            return {}
        template = spec.config_schema_class.model_json_schema()['$defs']
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

            # 处理 Union 类型
            origin = get_origin(param_type)
            is_union = origin is Union or isinstance(param_type, types.UnionType)
            if is_union and isinstance(value, dict):
                possible_types = get_args(param_type) if origin is Union else param_type.__args__
                type_name = value.get('type')
                if not type_name:
                    raise ValueError(f"Union field {name} in {cls.__name__} requires 'type' field")

                selected_cls = next((t for t in possible_types if inspect.isclass(t) and t.__name__ == type_name), None)
                if not selected_cls:
                    raise ValueError(
                        f"Unknown type '{type_name}' for {name}. Valid types: {[t.__name__ for t in possible_types if inspect.isclass(t)]}")

                sub_config = {k: v for k, v in value.items() if k != 'type'}
                kwargs[name] = self.__instantiate_with_config(selected_cls, sub_config)
            elif inspect.isclass(param_type) and hasattr(param_type, '__annotations__') and isinstance(value, dict):
                # 处理单个自定义类（不需要 type 字段）
                kwargs[name] = self.__instantiate_with_config(param_type, value)
            else:
                # 基本类型直接赋值
                kwargs[name] = value

        return cls(**kwargs)

    def __setting_default_value(self,spec: FunctionSpec, config: Dict[str, Any] | Any = None):
        if spec.default_setting != None:
            for key, value in spec.default_setting.items():
                config[key].update(value)
        return config

    def register_function(self, name: str, config: Dict[str, Any] | Any = None):
        if getattr(self, name) is not None:
            raise ValueError(f"[DevServer] Feature [{name}] already registered.")
        if name not in self.function_mapping:
            raise ValueError(f"[DevServer] Feature [{name}] not found in FUNCTION_MAPPING.")
        spec = self.function_mapping[name]
        config = self.__setting_default_value(spec, config)
        instance = self.__instantiate_with_config(spec.function_class, config) if config!={} else spec.function_class()
        setattr(self, name, instance)