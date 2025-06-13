import argparse
from .datamodel.configSchema import PromptOptimizeServerConfigArgs, OpenaiModelConfig, AgentConfig, \
    PromptOptimizeAgentConfig,PromptAnalyzeAgentConfig, PromptDetectAgentConfig, StructuredPromptEditAgentConfig

def setup_model() -> OpenaiModelConfig:
        parser = argparse.ArgumentParser(description='graph retriever setting')
        parser.add_argument("--base_url", type=str, default="https://api.rcouyi.com/v1")
        parser.add_argument("--api_key", type=str, default="sk-pAauG9ss64pQW9FVA703F1453b334eFb95B7447b9083BaBd")
        parser.add_argument("--model_name", type=str, default='gpt-4o')
        parser.add_argument("--system_prompt", type=str, default=None)
        return OpenaiModelConfig(**vars(parser.parse_args()))

def setup_agent() -> AgentConfig:
    parser = argparse.ArgumentParser(description='graph retriever setting')
    parser.add_argument("--model", default=setup_model())
    return AgentConfig(**vars(parser.parse_args()))




def setup_args():
        agent_args = setup_agent()

        prompt_detect_agent_config = PromptDetectAgentConfig(
            model=agent_args.model
        )

        prompt_analyze_agent_config = PromptAnalyzeAgentConfig(
            model=agent_args.model
        )

        structured_prompt_edit_config = StructuredPromptEditAgentConfig(
            model=agent_args.model
        )

        prompt_optimize_agent_config = PromptOptimizeAgentConfig(
            model = agent_args.model
        )

        return PromptOptimizeServerConfigArgs(
                prompt_detect_agent = prompt_detect_agent_config,
                prompt_analyze_agent = prompt_analyze_agent_config,
                structured_prompt_edit_agent = structured_prompt_edit_config,
                prompt_optimize_agent = prompt_optimize_agent_config
        )

args = setup_args()
