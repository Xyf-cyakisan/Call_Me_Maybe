from abc import ABC, abstractmethod

from torch import isin
from llm_sdk import Small_LLM_Model
from .models import Function
from typing import Any


class ParametersDecoder(ABC):
    @abstractmethod
    def constrained_decode(self, para: dict[str, Any], func: Function,
                           llm: Small_LLM_Model,
                           current_context: list[int], current_answer: list[
                               int]) -> tuple[bool, list[int]]:
        pass

    def _get_next_token_id(self, current_context: list[int],
                            current_answer: list[int], allowed_ids: list[int] | int,
                            llm: Small_LLM_Model) -> int:
        logits = llm.get_logits_from_input_ids(
            current_context + current_answer)
        if isinstance(allowed_ids, list):
            to_max = max([logits[allowed_id] for allowed_id in allowed_ids])
        else:
            to_max = logits[allowed_ids]
        next_token_id = logits.index(to_max)
        return next_token_id


class NumberDecoder(ParametersDecoder):
    NUMBERS_REGEX = ["0", "1", "2", "3", "4", "5", "6", "7",
                     "8", "9", "-", ".", "\n"]

    def constrained_decode(self, para: str, func: Function,
                           llm: Small_LLM_Model,
                           current_context: list[int], current_answer: list[
                               int]) -> tuple[bool, list[int]]:
        end = False
        para_name_ids = llm.encode("'" + para + "'")[0].tolist()
        for id in para_name_ids:
            current_answer.append(self._get_next_token_id(current_context,
                                  current_answer, id, llm))

        between_ids = llm.encode(": ")[0].tolist()
        for between_id in between_ids:
            current_answer.append(self._get_next_token_id(current_context,
                                  current_answer, id, llm))

        allowed_ids = [llm.encode(numb_char)[0].tolist()[0]
                       for numb_char in self.NUMBERS_REGEX]
        while True:
            current_answer.append(self._get_next_token_id(current_context,
                                  current_answer, allowed_ids, llm))
            if current_answer[-1] == '\n':
                current_answer = current_answer[:-1]
                break
        if list(func.parameters.keys())[-1] == para:
            end = True
        else:
            end_ids = llm.encode(", ")[0].tolist()
            for end_id in end_ids:
                current_answer.append(self._get_next_token_id(current_context,
                                      current_answer, end_id, llm))
        return (end, current_answer)


class StringDecoder(ParametersDecoder):
    def constrained_decode(self, para: dict[str, Any], func: Function,
                           llm: Small_LLM_Model,
                           current_context: list[int], current_answer: list[
                               int]) -> tuple[bool, list[int]]:
        ...


class BoolDecoder(ParametersDecoder):
    BOOL_REGEX = ["true", "false"]

    def constrained_decode(self, para: dict[str, Any], func: Function,
                           llm: Small_LLM_Model,
                           current_context: list[int], current_answer: list[
                               int]) -> tuple[bool, list[int]]:
        allowed_ids = [llm.encode(numb_char)[0].tolist()[0]
                       for numb_char in self.BOOL_REGEX]
