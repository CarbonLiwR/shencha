import argparse

from .data.prompt.extract import EXTRACT_ENTITIES_FROM_QUERY
from .datamodel.configSchema import GraphServerConfigArgs, GraphRetrieverConfig, ContextBuilderConfig, RankerConfig, SubGraphExtractorConfig, EmbeddingModelConfig, ExtractModelConfig
from .datamodel.model_output import EntityExtractedRes


def setup_extract_model() -> ExtractModelConfig:
        parser = argparse.ArgumentParser(description='graph retriever setting')
        parser.add_argument("--base_url", type=str, default="https://api.rcouyi.com/v1")
        parser.add_argument("--api_key", type=str, default="sk-pAauG9ss64pQW9FVA703F1453b334eFb95B7447b9083BaBd")
        parser.add_argument('--model_name', type=str, default="gpt-4o")
        parser.add_argument('--system_prompt', type=str, default=EXTRACT_ENTITIES_FROM_QUERY)
        parser.add_argument('--output_format', default=EntityExtractedRes)
        return ExtractModelConfig(**vars(parser.parse_args()))

def setup_embedding_model() -> EmbeddingModelConfig:
        parser = argparse.ArgumentParser(description='graph retriever setting')
        parser.add_argument("--base_url", type=str, default="https://api.rcouyi.com/v1")
        parser.add_argument("--api_key", type=str, default="sk-pAauG9ss64pQW9FVA703F1453b334eFb95B7447b9083BaBd")
        parser.add_argument("--model_name",type=str, default="text-embedding-3-small")
        return EmbeddingModelConfig(**vars(parser.parse_args()))

def setup_graph_extractor() -> SubGraphExtractorConfig:
        parser = argparse.ArgumentParser(description='graph retriever setting')
        parser.add_argument("--graph_level",type=int,default=None)
        return SubGraphExtractorConfig(**vars(parser.parse_args()))

def setup_ranker() -> RankerConfig:
        parser = argparse.ArgumentParser(description='graph retriever setting')
        parser.add_argument("--if_reverse", type=bool, default=False)
        return RankerConfig(**vars(parser.parse_args()))

def setup_context_builder() -> ContextBuilderConfig:
        parser = argparse.ArgumentParser(description='graph retriever setting')
        parser.add_argument("--max_context_tokens",type=int, default=8000)
        parser.add_argument("--ranker", default=setup_ranker())
        return ContextBuilderConfig(**vars(parser.parse_args()))


def setup_graph_semantic_retriever_args():
        parser = argparse.ArgumentParser(description='graph retriever setting')
        parser.add_argument("--extract_model",default=setup_extract_model())
        parser.add_argument("--embedding_model", default=setup_embedding_model())
        parser.add_argument('--sub_graph_extractor', default=setup_graph_extractor())
        parser.add_argument('--ranker', default=setup_ranker())
        parser.add_argument("--context_builder", default=setup_context_builder())
        return GraphRetrieverConfig(**vars(parser.parse_args()))


def setup_args():
        graph_retriever_args = setup_graph_semantic_retriever_args()

        graph_retriever_config = GraphRetrieverConfig(
                extract_model = graph_retriever_args.extract_model,
                embedding_model = graph_retriever_args.embedding_model,
                sub_graph_extractor = graph_retriever_args.sub_graph_extractor,
                ranker = graph_retriever_args.ranker,
                context_builder = graph_retriever_args.context_builder
        )


        return GraphServerConfigArgs(
                graph_retriever = graph_retriever_config
        )


graph_server_args = setup_args()
