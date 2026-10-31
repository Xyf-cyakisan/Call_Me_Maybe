from pydantic import BaseModel
from llm_sdk import Small_LLM_Model
from .input_models import Function
import math


class FunctionSelector(BaseModel):
    def __set_tokenized_funcs(self,
                              llm: Small_LLM_Model,
                              functions: list[Function]) -> None:
        self.__tokenized_funcs: dict[str, list[int]] = {
            func.name: llm.encode(func.name + '\n')[0].tolist()
            for func in functions}

    def get_func_name(self, llm: Small_LLM_Model,
                      context: str,
                      functions: list[Function]) -> str:

        self.__set_tokenized_funcs(llm, functions)
        tokenized_context = llm.encode(context)[0].tolist()
        tokenized_answer: list[int] = []
        while True:
            allowed_ids = []
            for tokened_func in self.__tokenized_funcs.values():
                if (len(tokened_func) > len(tokenized_answer)
                    and tokenized_answer == tokened_func[:len(tokenized_answer)
                                                         ]):
                    allowed_ids.append(tokened_func[len(tokenized_answer)])
            if not allowed_ids:
                raise ValueError("No valid token can continue a function"
                                 f"name. {llm.decode(tokenized_answer)}")
            elif len(allowed_ids) == 1 or len(set(allowed_ids)) == 1:
                next_token_id = allowed_ids[0]
            else:
                logits = llm.get_logits_from_input_ids(
                    tokenized_context + tokenized_answer
                )
                for token_id in allowed_ids:
                    if math.isnan(logits[token_id]):
                        raise ValueError("NaN logit encountered.")
                allowed_set = set(allowed_ids)
                for token_id in range(len(logits)):
                    if token_id not in allowed_set:
                        logits[token_id] = float("-inf")
                next_token_id = max(
                    range(len(logits)),
                    key=lambda token_id: logits[token_id]
                )
            tokenized_answer.append(next_token_id)
            if tokenized_answer in self.__tokenized_funcs.values():
                str_func = llm.decode(tokenized_answer)[:-1]
                for func in functions:
                    if func.name == str_func:
                        return str_func
