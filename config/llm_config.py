# C:\Users\1\Desktop\shencha-master\config\llm_config.py
import os
from dotenv import load_dotenv

# 加载 .env 文件
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env1')
load_dotenv(env_path)


class LLMConfig:
    """LLM 相关配置"""

    @property
    def api_url(self) -> str:
        return os.getenv('LLM_API_URL', 'https://api.rcouyi.com/v1/chat/completions')

    @property
    def model_name(self) -> str:
        return os.getenv('LLM_MODEL_NAME', 'gpt-4o')

    @property
    def api_key(self) -> str:
        key = os.getenv('LLM_API_KEY')
        if not key:
            raise ValueError("LLM_API_KEY 未在环境变量中设置")
        return key


# 创建全局配置实例
llm_config = LLMConfig()
