import inspect
from typing import Dict, Any, Type
from .component.ranker import FieldRanker
from .component.modelcall import EmbeddingModel, ChatModel
from .component.context_builder import GraphContextBuilder
from .component.graph_extractor import SubGraphExtractor
from .component.graph_retriever import GraphRetriever
from .component.loader import GraphLoader
from .config import graph_server_args
from .base.server import KnowledgeGraphServer
from .datamodel.configSchema import FunctionSpec, GraphRetrieverConfig
from .data.prompt.extract import EXTRACT_ENTITIES_FROM_QUERY
from .datamodel.model_output import EntityExtractedRes


class KGraphUserServer(KnowledgeGraphServer):
    def __init__(self):
        super(KGraphUserServer, self).__init__()
        try:
            if self.graph_retriever == None \
                    or self.graph_loader == None:
                self.__init_server()
            else:
                pass
        except Exception as e:
            raise RuntimeError(f"[KnowledgeGraphServer] Initialization failed: {str(e)}")

    def __init_server(self) -> None:
        self.graph_retriever = GraphRetriever(
                    extract_model = graph_server_args.graph_retriever.extract_model,
                    embedding_model= graph_server_args.graph_retriever.embedding_model,
                    sub_graph_extractor= graph_server_args.graph_retriever.sub_graph_extractor,
                    ranker= graph_server_args.graph_retriever.ranker,
                    context_builder=graph_server_args.graph_retriever.context_builder
                )
        self.graph_loader = GraphLoader()
class KGraphDevServer(KnowledgeGraphServer):
    def __init__(self):
        super(KGraphDevServer, self).__init__()
        self.function_mapping = {
            "graph_retriever": FunctionSpec(
                function_class=GraphRetriever,
                config_schema_class=GraphRetrieverConfig,
                default_setting = {
                    "extract_model": {
                        "system_prompt": EXTRACT_ENTITIES_FROM_QUERY,
                        "output_format": EntityExtractedRes
                    }
                }
            ),
            "graph_loader":FunctionSpec(
                function_class=GraphLoader,
                config_schema_class=None,
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




