from abc import ABC, abstractmethod
import json
import re
from llm_sdk import Small_LLM_Model
from .models import Function


class ParametersDecoder(ABC):
    @abstractmethod
    def constrained_decode(cls, para: str, func: Function, prompt: str,
                           llm: Small_LLM_Model,
                           current_cont: list[int], current_answer: list[
                               int]) -> tuple[bool, list[int]]:
        pass

    @classmethod
    def _set_name(cls, llm: Small_LLM_Model, para: str,
                  current_cont: str, current_answer: str):
        para_name_ids = llm.encode('"' + para + '"')[0].tolist()
        for id in para_name_ids:
            current_answer.append(cls._get_next_token_id(current_cont,
                                  current_answer, id, llm))

        between_ids = llm.encode(": ")[0].tolist()
        for between_id in between_ids:
            current_answer.append(cls._get_next_token_id(current_cont,
                                  current_answer, between_id, llm))
        return current_answer

    @classmethod
    def _get_next_token_id(cls, current_context: list[int],
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
    @classmethod
    def __extract_numbers(cls, prompt: str, llm: Small_LLM_Model) -> list[int]:
        return [llm.encode(number + '\n')[0].tolist()
                for number in
                re.findall(r"-?\d+(?:\.\d+)?",
                prompt)]

    @classmethod
    def constrained_decode(cls, values: dict[str, object],
                           para: str, func: Function, prompt: str,
                           llm: Small_LLM_Model, current_cont: list[int],
                           current_answer: list[
                               int]) -> tuple[bool, list[int]]:
        end = False
        cls._set_name(llm, para, current_cont, current_answer)
        allowed_numbers_id = cls.__extract_numbers(prompt, llm)
        str_number = []
        while True:
            allowed_ids = []
            for number_id in allowed_numbers_id:
                if (len(number_id) >= len(str_number)
                   and str_number == number_id[:len(str_number)]):
                    allowed_ids.append(number_id[len(str_number)])
            str_number.append(cls._get_next_token_id(current_cont,
                                                     str_number,
                                                     allowed_ids, llm))
            if (llm.decode([str_number[-1]]) == "\n"
               and str_number in allowed_numbers_id):
                current_answer += str_number[:-1]
                break
        print(llm.decode(current_answer) + "}")
        values.update(json.loads(llm.decode(current_answer) + "}"))
        if list(func.parameters.keys())[-1] == para:
            end = True
        else:
            end_ids = llm.encode(", ")[0].tolist()
            for end_id in end_ids:
                current_answer.append(cls._get_next_token_id(current_cont,
                                      current_answer, end_id, llm))
        return (end, current_answer, values)


class StringDecoder(ParametersDecoder):
    @classmethod
    def __extract_allowed_ids(cls) -> None:
        return [number for number in range(151927)]

    @classmethod
    def constrained_decode(cls, values: dict[str, object],
                           para: str, func: Function, prompt: str,
                           llm: Small_LLM_Model, current_cont: list[int],
                           current_answer: list[
                               int]) -> tuple[bool, list[int]]:
        end = False
        cls._set_name(llm, para, current_cont, current_answer)
        allowed_ids = cls.__extract_allowed_ids()
        string = []
        string.append(cls._get_next_token_id(current_cont,
                                             current_answer,
                                             llm.encode('"')[0].tolist(),
                                             llm))
        while True:
            string.append(cls._get_next_token_id(current_cont,
                                                 string,
                                                 allowed_ids,
                                                 llm))
            if ("\n" in llm.decode(string[-1])):
                string = llm.decode(string)[:-1]
                current_answer += llm.encode(string)[0].tolist()
                break
        decoded = llm.decode(current_answer) + "}"
        print(decoded)
        values.update(json.loads(decoded))
        if list(func.parameters.keys())[-1] == para:
            end = True
        else:
            end_ids = llm.encode(", ")[0].tolist()
            for end_id in end_ids:
                current_answer.append(cls._get_next_token_id(current_cont,
                                      current_answer, end_id, llm))
        return (end, current_answer, values)


class BoolDecoder(ParametersDecoder):
    BOOL_REGEX = ["true", "false"]

    @classmethod
    def constrained_decode(cls, values: dict[str, object],
                           para: str, func: Function, prompt: str,
                           llm: Small_LLM_Model, current_cont: list[int],
                           current_answer: list[
                               int]) -> tuple[bool, list[int]]:
        allowed_ids = [llm.encode(numb_char)[0].tolist()[0]
                       for numb_char in cls.BOOL_REGEX]
