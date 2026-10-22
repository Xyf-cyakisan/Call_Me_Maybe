from abc import ABC, abstractmethod
import re
from llm_sdk import Small_LLM_Model
from .models import Function


class ParametersDecoder(ABC):
    @abstractmethod
    def constrained_decode(self, para: str, func: Function, prompt: str,
                           llm: Small_LLM_Model,
                           current_context: list[int], current_answer: list[
                               int]) -> tuple[bool, list[int]]:
        pass

    def _get_next_token_id(self, current_context: list[int],
                           current_answer: list[int], allowed_ids: list[int]
                           | int,
                           llm: Small_LLM_Model) -> int:
        logits = llm.get_logits_from_input_ids(
            current_context + current_answer
        )

        if isinstance(allowed_ids, list):
            return max(
                allowed_ids,
                key=lambda token_id: logits[token_id],
            )

        return allowed_ids


class NumberDecoder(ParametersDecoder):
    def __extract_numbers(self, prompt: str, llm: Small_LLM_Model) -> None:
        try:
            self.__allowed_numbers_id
        except AttributeError:
            self.__allowed_numbers_id = [llm.encode(number + '\n')[0].tolist()
                                         for number in
                                         re.findall(r"-?\d+(?:\.\d+)?",
                                                    prompt)]
        else:
            return

    def constrained_decode(self, para: str, func: Function, prompt: str,
                           llm: Small_LLM_Model,
                           current_context: list[int], current_answer: list[
                               int]) -> tuple[bool, list[int]]:
        end = False
        para_name_ids = llm.encode('"' + para + '"')[0].tolist()
        for id in para_name_ids:
            current_answer.append(self._get_next_token_id(current_context,
                                  current_answer, id, llm))

        between_ids = llm.encode(": ")[0].tolist()
        for between_id in between_ids:
            current_answer.append(self._get_next_token_id(current_context,
                                  current_answer, between_id, llm))

        self.__extract_numbers(prompt, llm)
        str_number = []
        while True:
            allowed_ids = []
            for number_id in self.__allowed_numbers_id:
                if (len(number_id) >= len(str_number)
                   and str_number == number_id[:len(str_number)]):
                    allowed_ids.append(number_id[len(str_number)])
            str_number.append(self._get_next_token_id(current_context,
                              str_number, allowed_ids, llm))
            if llm.decode([str_number[-1]]) == "\n":
                current_answer += str_number[:-1]
                break
        self.__allowed_numbers_id.remove(str_number)
        if list(func.parameters.keys())[-1] == para:
            end = True
        else:
            end_ids = llm.encode(", ")[0].tolist()
            for end_id in end_ids:
                current_answer.append(self._get_next_token_id(current_context,
                                      current_answer, end_id, llm))
        return (end, current_answer)


class StringDecoder(ParametersDecoder):
    def constrained_decode(self, para: str, func: Function, prompt: str,
                           llm: Small_LLM_Model,
                           current_context: list[int], current_answer: list[
                               int]) -> tuple[bool, list[int]]:
        ...


class BoolDecoder(ParametersDecoder):
    BOOL_REGEX = ["true", "false"]

    def constrained_decode(self, para: str, func: Function, prompt: str,
                           llm: Small_LLM_Model,
                           current_context: list[int], current_answer: list[
                               int]) -> tuple[bool, list[int]]:
        allowed_ids = [llm.encode(numb_char)[0].tolist()[0]
                       for numb_char in self.BOOL_REGEX]
