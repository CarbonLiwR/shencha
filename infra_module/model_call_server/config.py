import argparse
from .datamodel.model_output import ModelRes
from .datamodel.configSchema import ModelCallServerConfigArgs, LocalEmbedModelConfig, OpenaiEmbedModelConfig, OpenaiChatModelConfig

def setup_openai_model():
        parser = argparse.ArgumentParser(description='graph retriever setting')
        parser.add_argument("--base_url", type=str, default="https://api.rcouyi.com/v1")
        parser.add_argument("--api_key", type=str, default="sk-pAauG9ss64pQW9FVA703F1453b334eFb95B7447b9083BaBd")
        parser.add_argument("--embedding_model_name", type=str, default="text-embedding-3-small")
        parser.add_argument("--chat_model_name", type=str, default='gpt-4o')
        parser.add_argument("--chat_model_prompt", type=str, default='You are a chat assistant')
        parser.add_argument("--chat_model_output_format",default=ModelRes)

        return parser.parse_args()

def setup_local_model():
        parser = argparse.ArgumentParser(description='graph retriever setting')
        parser.add_argument("--local_embedding_model_path", type=str, default="E:/publicLib/infra_module/model_call_server/data/model")
        return parser.parse_args()


def setup_args():
        openai_args = setup_openai_model()
        local_args = setup_local_model()

        openai_chat_model_config = OpenaiChatModelConfig(
                base_url=openai_args.base_url,
                api_key=openai_args.api_key,
                model_name=openai_args.chat_model_name,
                system_prompt=openai_args.chat_model_prompt,
                output_format=openai_args.chat_model_output_format

        )
        openai_embedding_model_config = OpenaiEmbedModelConfig(
                base_url=openai_args.base_url,
                api_key=openai_args.api_key,
                model_name=openai_args.embedding_model_name
        )

        # Get local config
        local_embedding_model_config = LocalEmbedModelConfig(
                model_path=local_args.local_embedding_model_path
        )

        return ModelCallServerConfigArgs(
                openai_chat_model=openai_chat_model_config,
                openai_embedding_model=openai_embedding_model_config,
                local_embedding_model=local_embedding_model_config
        )


args = setup_args()
