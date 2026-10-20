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
            context += "Return only the function name.\n"
            context += "Function:"
            yield context

    def __set_llm(self) -> None:
        self.__llm = Small_LLM_Model()

    def __set_tokenized_funcs(self) -> None:
        self.__tokenized_funcs: dict[str, list[int]] = {
            func.name: self.__llm.encode(func.name)[0].tolist()
            for func in self.functions}

    def run_llm(self) -> None:
        self.__set_llm()
        self.__set_tokenized_funcs()
        for i, context in enumerate(self.__build_contexts(), start=1):
            tokenized_context = self.__llm.encode(context)[0].tolist()
            tokenized_answer = []
            print(f"Context {i}")
            print(context)
            while True:
                allowed_ids = []
                logits = self.__llm.get_logits_from_input_ids(
                        tokenized_context + tokenized_answer
                )
                for tokened_func in self.__tokenized_funcs.values():
                    if (len(tokened_func) >= len(tokenized_answer)
                        and tokenized_answer == tokened_func[:len(tokenized_answer)]):
                        allowed_ids.append(tokened_func[len(tokenized_answer)])
                next_token_id = logits.index((max([logits[allowed_id]
                                                   for allowed_id in
                                                   allowed_ids])))
                tokenized_answer.append(next_token_id)
                if tokenized_answer in self.__tokenized_funcs.values():
                    print(f"Answer: {self.__llm.decode(tokenized_answer)}\n\n")
                    break
