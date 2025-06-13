import inspect
from pathlib import Path
from typing import Dict, Any, Type
from .component.data_loader import DataLoader
from .component.extractor import TableExtractor, TextExtractor, ImageExtractor
from .component.chunker import Chunker
from .base.server import DoclingServer
from .datamodel.configSchema import DataChunkerConfig, DataLoaderConfig, TableExtractorConfig, ImageExtractorConfig, TextExtractorConfig, FunctionSpec
from .config import args

class DoclingUserServer(DoclingServer):
    def __init__(self):
        super(DoclingUserServer, self).__init__()
        try:
            if self.data_loader == None or \
                    self.text_extractor == None or \
                    self.image_extractor == None or \
                    self.table_extractor == None or \
                    self.data_chunker == None:
                self.__init_server()
            else:
                pass
        except Exception as e:
            raise RuntimeError(f"[DoclingServer] Initialization failed: {str(e)}")

    def __init_server(self) -> None:
        self.data_loader = DataLoader()
        self.text_extractor = TextExtractor()
        self.image_extractor = ImageExtractor(
            save_path= Path(args.image_extractor.save_path)
        )
        self.table_extractor = TableExtractor(
            save_type=args.table_extractor.save_type,
            save_path=Path(args.table_extractor.save_path)
        )
        self.data_chunker = Chunker(
            model_name=args.data_chunker.model_name,
            max_tokens=args.data_chunker.max_tokens
        )

class DoclingDevServer(DoclingServer):
    def __init__(self):
        super(DoclingDevServer, self).__init__()
        self.function_mapping = {
            "data_loader": FunctionSpec(
                function_class=DataLoader,
                config_schema_class=None,
                default_setting=None
            ),
            "text_extractor": FunctionSpec(
                function_class=TableExtractor,
                config_schema_class=TextExtractorConfig,
                default_setting=None
            ),
            "image_extractor":FunctionSpec(
                function_class=ImageExtractor,
                config_schema_class=ImageExtractorConfig,
                default_setting=None
            ),
            "table_extractor":FunctionSpec(
                function_class=TableExtractor,
                config_schema_class=TableExtractorConfig,
                default_setting=None
            ),
            "data_chunker":FunctionSpec(
                function_class=Chunker,
                config_schema_class=DataChunkerConfig,
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