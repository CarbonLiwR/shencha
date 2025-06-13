


from abc import ABC, abstractmethod


class PromptOptimizeServer(ABC):
    def __init__(self):
        self.prompt_detect_agent = None
        self.prompt_analyze_agent = None
        self.sprompt_edit_agent = None
        self.prompt_optimize_agent = None

    def detect_prompt(self, prompt):
        detected_result = self.prompt_detect_agent.detect(prompt)
        return detected_result

    def analyze_prompt(self, prompt):
        analyzed_prompt = self.prompt_analyze_agent.analyze(prompt)
        return analyzed_prompt

    def edit_structured_prompt(self, prompt, analysis_result):
        structured_prompt = self.sprompt_edit_agent.edit(prompt, analysis_result)
        return structured_prompt

    def optimize_prompt(self, prompt):
        optimized_prompt = self.prompt_optimize_agent.optimize(prompt)
        return optimized_prompt





