from typing import Generator
from pydantic import BaseModel
from .models import Function, Prompt
from llm_sdk import Small_LLM_Model


class LLMOrchestrator(BaseModel):
    functions: list[Function]
    prompts: list[Prompt]

    def __build_context(self) -> Generator[str]:
        for prompt in self.prompts:
            context = ""
            for i, func in enumerate(self.functions):
                context += f"Function {i}: {func.name}\n"
                context += f"Description {i}: {func.description}\n\n"
            context += f"Prompt: {prompt.prompt}\n\n"
            context += "Choose the correct function to resolve the prompt."
            yield context

    def __set_llm(self) -> None:
        self.__llm = Small_LLM_Model()

    def run_llm(self) -> None:
        self.__set_llm()
        contexts = self.__build_context()
        for context in contexts:
            inputs_ids = self.__llm.encode(context)
            print(self.__llm.get_logits_from_input_ids(inputs_ids))
