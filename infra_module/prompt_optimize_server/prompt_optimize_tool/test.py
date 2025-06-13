from prompt_optimize_tool import PromptOptimizeTool
import asyncio

POT = PromptOptimizeTool(
    {"key": "sk-qaYkcMpKWCiAqr5f5369063175Ef4d65A756Ae7e287e11Bb", "url": "https://api.rcouyi.com/v1"})
a = asyncio.run(POT.main(
    "帮我写一篇小学生清明节春游踏青的周记，结合传统文化，以小学生的口吻描述所见所感，不超过五百个字。"))
print(a)