import asyncio
import json
import logging
import subprocess
import time
from pathlib import Path
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters

def load_server_config(config_file: str):
    """从配置文件中加载服务器信息"""
    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件 {config_file} 不存在")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)

def start_servers(config: dict):
    """
    自动启动多个 MCP 服务器进程
    """
    processes = {}
    for srv in config.get("servers", []):
        proc = subprocess.Popen([srv["command"]] + srv["args"])
        processes[srv["name"]] = proc
        # 给服务器一些启动时间
        time.sleep(1)
    return processes

def stop_servers(processes: dict):
    """终止所有服务器进程并确保退出"""
    for name, proc in processes.items():
        try:
            proc.terminate()
            # 等待子进程终止，最多 2 秒
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            logging.warning(f"进程 {name} 未正常退出，强制终止")
            proc.kill()
            proc.wait()
        except Exception as e:
            logging.error(f"终止进程 {name} 失败: {str(e)}")

async def call_server(tool_name: str, args: dict, server_info: dict):
    """
    利用 stdio_client 连接单个服务器，并调用工具
    """
    server_params = StdioServerParameters(
        command=server_info["command"],
        args=server_info["args"]
    )
    async with stdio_client(server_params) as (stdio, write):
        async with ClientSession(stdio, write) as session:
            await session.initialize()
            response = await session.call_tool(tool_name, args)
            return response

class PromptOptimizeTool:
    def __init__(self, config: dict):
        self.key = config["key"]
        self.url = config["url"]

    async def run(self, user_prompt: str):
        config = load_server_config(str(Path(__file__).parent / "prompt_optimize_tool_module/servers_config.json"))
        server_processes = start_servers(config)
        optimized_user_prompt = user_prompt  # 初始化变量

        try:
            agent_server_info = next(srv for srv in config["servers"] if srv["name"] == "PromptOptimizeServer")
            # 工具调用
            tool_result = await call_server("prompt_optimize_tool", {"key": self.key, "url": self.url, "user_prompt": user_prompt}, agent_server_info)
            optimized_user_prompt = tool_result.content[0].text.strip()

        except Exception as e:
            logging.error(f"处理失败: {str(e)}")
            yield {"type": "error", "message": str(e)}
            raise

        finally:
            stop_servers(server_processes)
            await asyncio.sleep(0.1)

        yield optimized_user_prompt

    async def main(self, user_prompt: str):
        try:
            async for result in self.run(user_prompt):
                return result
        except Exception as e:
            return f"优化失败: {str(e)}"
        finally:
            # 等待事件循环清理残留任务
            await asyncio.sleep(0)

if __name__ == '__main__':
    POT = PromptOptimizeTool({"key": "sk-qaYkcMpKWCiAqr5f5369063175Ef4d65A756Ae7e287e11Bb", "url": "https://api.rcouyi.com/v1"})
    a = asyncio.run(POT.main("帮我写一篇面向对象为非计算机专业的大学生的文章，主题为人工智能科普，体裁为新闻稿。要求五千字以内中文自然段，使用纯文本输出。"))
    print(a)