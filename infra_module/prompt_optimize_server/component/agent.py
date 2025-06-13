from .model_call import OpenaiChatModel
from ..data import prompt


class DetectAgent():
    def __init__(self, model: OpenaiChatModel):
        self.model: OpenaiChatModel = model
        self.steps = ["detecting_trap"]

    def __get_system_prompt_by_step(self, step_name):
        system_prompt = getattr(prompt, step_name)
        return system_prompt

    def detect(self, prompt):
        user_massage = f"需要优化的用户提示语：\n{prompt}"
        result = ""
        for step in self.steps:
            self.model.system_prompt = self.__get_system_prompt_by_step(step)
            res = self.model.generate(user_massage)
            result+=f"\n{res}"
        return result

class AnalysisAgent():
    def __init__(self, model: OpenaiChatModel):
        self.model: OpenaiChatModel = model
        self.steps = ["anchoring_target", "activate_role", "disassembly_task", "expand_thinking", "focus_subject"]

    def __get_system_prompt_by_step(self, step_name):
        system_prompt = getattr(prompt, step_name)
        return system_prompt

    def analyze(self, prompt):
        user_massage = f"用户提示语：\n{prompt}"
        result = ""
        for step in self.steps:
            self.model.system_prompt = self.__get_system_prompt_by_step(step)
            res = self.model.generate(user_massage)
            result += f"\n{res}"
        return result


class StructurePromptEditAgent():
    def __init__(self, model):
        self.model = model
        self.steps = ["structuring_prompt"]

    def __get_system_prompt_by_step(self, step_name):
        system_prompt = getattr(prompt, step_name)
        return system_prompt

    def edit(self, prompt, analysis_result):
        user_massage = f"用户提示语：\n{prompt}\n用户提示语分析结果：{analysis_result}"
        result = ""
        for step in self.steps:
            self.model.system_prompt = self.__get_system_prompt_by_step(step)
            res = self.model.generate(user_massage)
            result += f"\n{res}"
        return result

class OptimizeAgent():
    def __init__(self, model):
        self.model = model
        self.steps = ["review_progressive", "verification_logic", "balance_focus", "optimizing_representation"]

    def __get_system_prompt_by_step(self, step_name):
        system_prompt = getattr(prompt, step_name)
        return system_prompt

    def optimize(self, prompt):
        user_massage = f"需要优化的用户提示语：\n{prompt}"
        res = None
        for step in self.steps:
            self.model.system_prompt = self.__get_system_prompt_by_step(step)
            res = self.model.generate(user_massage)
        return res


