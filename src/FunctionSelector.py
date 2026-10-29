from pydantic import BaseModel
from llm_sdk import Small_LLM_Model
from .input_models import Function


class FunctionSelector(BaseModel):
    def __set_tokenized_funcs(self,
                              llm: Small_LLM_Model,
                              functions: list[Function]) -> str:
        self.__tokenized_funcs: dict[str, list[int]] = {
            func.name: llm.encode(func.name + '\n')[0].tolist()
            for func in functions}

    def get_func_name(self, llm: Small_LLM_Model,
                      context: str,
                      functions: list[Function]) -> str:

        self.__set_tokenized_funcs(llm, functions)
        tokenized_context = llm.encode(context)[0].tolist()
        tokenized_answer = []
        while True:
            allowed_ids = []
            logits = llm.get_logits_from_input_ids(
                    tokenized_context + tokenized_answer
            )
            for tokened_func in self.__tokenized_funcs.values():
                if (len(tokened_func) >= len(tokenized_answer)
                    and tokenized_answer == tokened_func[:len(tokenized_answer)
                                                         ]):
                    allowed_ids.append(tokened_func[len(tokenized_answer)])
            next_token_id = logits.index((max([logits[allowed_id]
                                               for allowed_id in
                                               allowed_ids])))
            tokenized_answer.append(next_token_id)
            if tokenized_answer in self.__tokenized_funcs.values():
                str_func = llm.decode(tokenized_answer[:-1])
                for func in functions:
                    if func.name == str_func:
                        return str_func
