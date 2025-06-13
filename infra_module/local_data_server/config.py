import argparse
from .datamodel.configSchema import LocalDataServerConfigArgs, SemanticRetrieverConfig,RankerConfig,RegexChunkerConfig,DataLoaderConfig,LocalEmbedModelConfig



def setup_local_model() -> LocalEmbedModelConfig:
        parser = argparse.ArgumentParser(description='graph retriever setting')
        parser.add_argument("--model_path", type=str, default="E:/publicLib/infra_module/model_call_server/data/model")
        return LocalEmbedModelConfig(**vars(parser.parse_args()))


def setup_data_retriever() -> SemanticRetrieverConfig:
        parser = argparse.ArgumentParser(description='graph retriever setting')
        parser.add_argument("--embedding_model", default=setup_local_model())
        return SemanticRetrieverConfig(**vars(parser.parse_args()))


def setup_data_ranker() -> RankerConfig:
        parser = argparse.ArgumentParser(description='graph retriever setting')
        parser.add_argument("--if_reverse", type=bool, default=True)
        return RankerConfig(**vars(parser.parse_args()))

def setup_regex_chunker() -> RegexChunkerConfig:
        parser = argparse.ArgumentParser(description='graph retriever setting')
        parser.add_argument("--regex_pattern",type=str,default=r"\n")
        return RegexChunkerConfig(**vars(parser.parse_args()))


def setup_args() -> LocalDataServerConfigArgs:
        data_retriever_args = setup_data_retriever()
        data_ranker_args = setup_data_ranker()
        data_regex_chunker = setup_regex_chunker()

        data_retriever_config = SemanticRetrieverConfig(
                embedding_model= data_retriever_args.embedding_model
        )
        data_chunker_config = RegexChunkerConfig(
                regex_pattern = data_regex_chunker.regex_pattern
        )
        data_ranker_config = RankerConfig(
                if_reverse = data_ranker_args.if_reverse
        )
        data_loader_config = DataLoaderConfig()

        return LocalDataServerConfigArgs(
                data_retriever = data_retriever_config,
                data_chunker = data_chunker_config,
                data_loader = data_loader_config,
                data_ranker = data_ranker_config
       )


args = setup_args()
