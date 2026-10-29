from llm_sdk import Small_LLM_Model
from .input_models import Function
from pydantic import BaseModel
from enum import Enum


class SyntaxStatus(Enum):
    START = 1
    EXPECTING_KEY = 2
    EXPECTING_COLON = 3
    EXPECTING_VALUE = 4
    EXPECTING_SEPARATOR = 5
    COMPLETE = 6


class SyntaxChecker(BaseModel):
    def set_values_at_beginning(self,
                                llm: Small_LLM_Model,
                                func: Function) -> None:
        self.__all_ids = [number for number in range(151927)]
        self.__llm = llm
        self.__func = func
        self.__status = SyntaxStatus.START
        self.__len_para = len(self.__func.parameters.keys())
        self.__parameters_count = 0
        self.__current_value = ""

    def __get_allowed_ids(self) -> list[int]:
        if self.__status == SyntaxStatus.START:
            return self.__llm.encode('{"')[0].tolist()
        if self.__status == SyntaxStatus.EXPECTING_COLON:
            return self.__llm.encode(': ')[0].tolist()
        if self.__status == SyntaxStatus.COMPLETE:
            return self.__llm.encode('}')[0].tolist()

    def __get_status(self, answer: str) -> int:
        key_or_value = False
        pot_status = SyntaxStatus.START
        in_string = False
        for char in answer:
            if char == '"':
                in_string = not in_string
                if in_string:
                    key_or_value = not key_or_value
                    if key_or_value and self.__func.parameters[self.__parameters_count].type == "string":
                        pot_status = SyntaxStatus.EXPECTING_VALUE
                    else:
                        pot_status = SyntaxStatus.EXPECTING_KEY
                else:
                    if key_or_value:
                        pot_status = SyntaxStatus.EXPECTING_COLON
                    else:
                        pot_status = SyntaxStatus.EXPECTING_SEPARATOR
        return pot_status

    def __get_highest_scored_token_id(self, current_context: list[int],
                                      tokenized_answer: list[int],
                                      token_ids: list[int] | int) -> int:
        logits = self.__llm.get_logits_from_input_ids(
            current_context + tokenized_answer
        )
        if isinstance(token_ids, list):
            return max(
                token_ids,
                key=lambda token_id: logits[token_id],
            )
        return token_ids

    def get_next_token_id(self, current_context: list[int],
                          tokenized_answer: list[int] | int
                          ) -> int | list[int]:
        allowed_ids = self.__get_allowed_ids()
        candidate = self.__get_highest_scored_token_id(
            current_context,
            tokenized_answer,
            allowed_ids)
        self.__change_status(tokenized_answer + [candidate])
        return candidate


    def is_complete(self, tokenized_answer: list[int]) -> bool:
        ...


class ParametersGenerator(BaseModel):
    syntax_checker: SyntaxChecker

    def generate(self,
                 func: Function,
                 llm: Small_LLM_Model,
                 tokenized_context: list[int]
                 ) -> tuple[bool, list[int]]:
        self.syntax_checker.set_values_at_beginning(llm, func)
        tokenized_answer = []
        while True:
            next_token_id = self.syntax_checker.get_next_token_id(
                                                     tokenized_context,
                                                     tokenized_answer)
            if isinstance(next_token_id, list):
                for token_id in next_token_id:
                    tokenized_answer.append(token_id)
            else:
                tokenized_answer.append(next_token_id)
            if self.syntax_checker.is_complete(tokenized_answer):
                break
        return (tokenized_answer)
