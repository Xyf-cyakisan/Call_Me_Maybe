from typing import Generator
from pydantic import BaseModel
from .models import Function, Prompt
from llm_sdk import Small_LLM_Model


class LLMOrchestrator(BaseModel):
    functions: list[Function]
    prompts: list[Prompt]

    def __build_contexts(self) -> Generator[str]:
        for prompt in self.prompts:
            context = ""
            for i, func in enumerate(self.functions, start=1):
                context += f"Function {i}: {func.name}\n"
                context += f"Description {i}: {func.description}\n\n"
            context += f"Prompt: {prompt.prompt}\n\n"
            context += ("Select the most appropriate function from the"
                        "list above.\n")
            context += "Return only the function name."
            context += "Function:"
            yield context

    def __set_llm(self) -> None:
        self.__llm = Small_LLM_Model()

    def __set_tokenized_funcs(self) -> None:
        self.__tokenized_funcs: dict[str, list[int]] = {
            func.name: self.__llm.encode(func.name) for func in self.functions}

    def run_llm(self) -> None:
        self.__set_llm()
        self.__set_tokenized_funcs()
        for context in self.__build_contexts():
            tokenized_context = self.__llm.encode(context)[0].tolist()
            tokenized_answer = []
            while True:
                logits = self.__llm.get_logits_from_input_ids(
                        tokenized_context + tokenized_answer
                )
                print(context)
                print([
                    (self.__llm.decode([token_id]), int(logits[token_id]))
                    for token_id in top_tokens
                ])
                if self.__llm.decode(tokenized_answer) in self.__tokenized_funcs.keys():
                    print(f"Answer: {self.__llm.decode(tokenized_answer)}\n\n")
                    break
