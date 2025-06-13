from mcp.server.fastmcp import FastMCP
import prompt as agent_prompt
from llm_util import OpenAIHandle

# 初始化 MCP 服务器，服务名称为 "PromptOptimizeServer"
mcp = FastMCP("PromptOptimizeServer")

@mcp.tool()
async def prompt_optimize_tool(key: str, url: str, user_prompt: str) -> str:
    ai_server = OpenAIHandle(key, url)
    system_message = agent_prompt.detecting_trap
    user_massage = f"需要优化的用户提示语：\n{user_prompt}"
    user_prompt = ai_server.answer_llm(system_message, user_massage)

    analysis_agents = ["anchoring_target", "activate_role", "disassembly_task", "expand_thinking", "focus_subject"]
    analysis_results = []
    for analysis_agent_name in analysis_agents:
        system_message = getattr(agent_prompt, analysis_agent_name)
        user_massage = f"用户提示语：\n{user_prompt}"
        analysis_result = ai_server.answer_llm(system_message, user_massage)
        analysis_results.append(analysis_result)

    system_message = agent_prompt.structuring_prompt
    user_massage = f"用户提示语：\n{user_prompt}\n用户提示语分析结果：{analysis_results}"
    user_prompt = ai_server.answer_llm(system_message, user_massage)

    optimize_agents = ["review_progressive", "verification_logic", "balance_focus", "optimizing_representation"]
    for optimize_agent_name in optimize_agents:
        system_message = getattr(agent_prompt, optimize_agent_name)
        user_massage = f"需要优化的用户提示语：\n{user_prompt}"
        user_prompt = ai_server.answer_llm(system_message, user_massage)

    return user_prompt

if __name__ == "__main__":
    # 使用标准输入输出传输启动 MCP 服务器
    mcp.run(transport="stdio")