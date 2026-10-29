import token
from typing import Any
from llm_sdk import Small_LLM_Model
from .input_models import Function
from pydantic import BaseModel
import json
import re


class ParametersGenerator(BaseModel):
    def __get_numbers_from_prompt(self, prompt: str) -> list[str]:
        return re.findall(r"-?\d+(?:\.\d+)?", prompt)

    def __get_integers_from_prompt(self, prompt: str) -> str:
        return re.findall(r"-?\d+", prompt)

    def __get_next_token_id(
        self,
        llm: Small_LLM_Model,
        current_context: list[int],
        tokenized_answer: list[int],
        token_ids: list[int],
    ):
        logits = llm.get_logits_from_input_ids(
            current_context + tokenized_answer
        )
        if isinstance(token_ids, list):
            return max(
                token_ids,
                key=lambda token_id: logits[token_id],
            )
        return token_ids

    def __check_escape_state(self,
                             str_answer: str,
                             allowed_chars: list[str]) -> str:
        escape_state = False
        for char in str_answer:
            if char == "\\":
                escape_state = not escape_state
            else:
                if escape_state:
                    if char in allowed_chars:
                        if char.lower() != "x" and char != "u":
                            escape_state = not escape_state
                else:
                    continue
        return escape_state

    def __escape_state(
        self, llm: Small_LLM_Model, tokenized_context: list[int],
        tokenized_answer: list[int], next_token_id: int
    ) -> str:
        if self.__check_if_string_end(llm, tokenized_answer,
                                      next_token_id):
            to_check = llm.decode(next_token_id)
            last_bit = to_check[:to_check.find('"')]
            next_token_id = llm.encode(last_bit)[0].tolist()
            for token_id in next_token_id:
                tokenized_answer.append(token_id)
            return "end"
        else:
            tokenized_answer.append(next_token_id)
        str_answer = llm.decode(tokenized_answer)
        allowed_chars = [
            '"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u'
        ]
        allowed_ids = [llm.encode(char)[0].tolist()[0]
                       for char in allowed_chars]
        allowed_ids += [token_id for token_id in range(151927)
                        if "\\" in llm.decode(token_id)]
        while self.__check_escape_state(str_answer, allowed_chars):
            tokenized_answer.append(self.__get_next_token_id(
                llm, tokenized_context, tokenized_answer,
                allowed_ids
            ))
            str_answer = llm.decode(tokenized_answer)
            if str_answer[-1].lower() == 'u':
                allowed_ids = [
                    "0", "1", "2", "3", "4", "5",
                    "6", "7", "8", "9"
                ]
                for _ in range(4):
                    tokenized_answer.append(self.__get_next_token_id(
                        llm, tokenized_context, tokenized_answer,
                        allowed_ids
                    ))
                str_answer = llm.decode(tokenized_answer)
            print(str_answer)
        return "continue"

    def __check_if_string_end(self, llm: Small_LLM_Model,
                              tokenized_answer: list[int],
                              next_token_id: int) -> bool:
        str_answer = llm.decode(tokenized_answer + [next_token_id])
        for i, char in enumerate(str_answer):
            if '"' in char:
                if (i == 0 or self.__check_escape_state(str_answer[:i + 1])
                   is False):
                    return True
                else:
                    continue
        return False

    def __get_string_value(
        self, llm: Small_LLM_Model, tokenized_context: list[int]
    ) -> str:
        true_context = tokenized_context + llm.encode('"')[0].tolist()
        tokenized_answer = []
        allowed_ids = [token_id for token_id in range(151927)]
        while True:
            next_token_id = self.__get_next_token_id(
                llm, true_context, tokenized_answer, allowed_ids
            )
            str_token = llm.decode(next_token_id)
            if "\\" in str_token:
                if self.__escape_state(llm, tokenized_context,
                                       tokenized_answer,
                                       next_token_id) == "end":
                    break
            elif '"' in str_token:
                if '"' in str_token:
                    to_check = str_token
                    last_bit = to_check[:to_check.find('"')]
                    next_token_id = llm.encode(last_bit)[0].tolist()
                    for token_id in next_token_id:
                        tokenized_answer.append(token_id)
                break
            else:
                tokenized_answer.append(next_token_id)
        raw_string = llm.decode(tokenized_answer)
        print(raw_string)
        return raw_string

    def __get_number_value(
        self, llm: Small_LLM_Model, tokenized_context: list[int]
    ) -> float:
        tokenized_answer = []
        while True:
            allowed_ids = llm.encode(",")[0].tolist()
            for integer_ids in self.__numbers_ids:
                if (len(tokenized_answer) < len(integer_ids)
                    and tokenized_answer == integer_ids[:len(tokenized_answer)
                                                        ]):
                    allowed_ids.append(integer_ids[len(tokenized_answer)])
            next_token_id = self.__get_next_token_id(
                llm, tokenized_context, tokenized_answer, allowed_ids
            )
            if "," in llm.decode(next_token_id):
                token = llm.decode(next_token_id)
                last_bit = token[: token.find(",")]
                next_token_id = llm.encode(last_bit)[0].tolist()
                for token_id in next_token_id:
                    tokenized_answer.append(token_id)
                break
            tokenized_answer.append(next_token_id)
        number_str = llm.decode(tokenized_answer)
        try:
            converted_number = int(number_str)
        except ValueError:
            converted_number = float(number_str)
        return converted_number

    def __get_boolean_value(
        self, llm: Small_LLM_Model, tokenized_context: list[int]
    ) -> float:
        tokenized_answer = []
        bools_ids = [llm.encode(value)[0].tolist() for value in ["true",
                                                                 "false"]]
        while True:
            allowed_ids = []
            for bool_id in bools_ids:
                if (len(tokenized_answer) < len(bool_id)
                   and tokenized_answer == bool_id[:len(tokenized_answer)]):
                    allowed_ids.append(bool_id[len(tokenized_answer)])
            next_token_id = self.__get_next_token_id(
                llm, tokenized_context, tokenized_answer, allowed_ids
            )
            tokenized_answer.append(next_token_id)
            if tokenized_answer in bools_ids:
                break
        return llm.decode(tokenized_answer) == "true"

    def __get_integer_value(
        self, llm: Small_LLM_Model, tokenized_context: list[int]
    ) -> float:
        tokenized_answer = []
        while True:
            allowed_ids = llm.encode(",")[0].tolist()
            for number_ids in self.__integers_ids:
                if (len(tokenized_answer) < len(number_ids)
                    and tokenized_answer == number_ids[:len(tokenized_answer)
                                                       ]):
                    allowed_ids.append(number_ids[len(tokenized_answer)])
            next_token_id = self.__get_next_token_id(
                llm, tokenized_context, tokenized_answer, allowed_ids
            )
            if "," in llm.decode(next_token_id):
                token = llm.decode(next_token_id)
                last_bit = token[: token.find(",")]
                next_token_id = llm.encode(last_bit)[0].tolist()
                for token_id in next_token_id:
                    tokenized_answer.append(token_id)
                break
            tokenized_answer.append(next_token_id)
        number_str = llm.decode(tokenized_answer)
        if int(number_str) == float(number_str):
            return int(number_str)
        else:
            return float(number_str)

    def generate(
        self,
        func: Function,
        prompt: str,
        llm: Small_LLM_Model,
        tokenized_context: list[int],
    ) -> dict[str, Any]:
        print(func.name, prompt)
        parameters = list(func.parameters.keys())
        dict_parameters = {}
        value_funcs = {
            "string": self.__get_string_value,
            "number": self.__get_number_value,
            "boolean": self.__get_boolean_value,
            "integer": self.__get_integer_value
        }
        values_types = [
            func.parameters[parameter].type for parameter in parameters
        ]
        if "number" in values_types or "integer" in values_types:
            if "number" in values_types:
                self.__numbers_ids = []
                for number in self.__get_numbers_from_prompt(prompt):
                    self.__numbers_ids.append(llm.encode(number)[0].tolist())
            if "integer" in values_types:
                self.__integers_ids = []
                for integer in self.__get_integers_from_prompt(prompt):
                    self.__integers_ids.append(llm.encode(integer)[0].tolist())
        for parameter in parameters:
            dict_parameters[parameter] = ""
            str_dict = json.dumps(dict_parameters)
            str_dict = str_dict[:-3]
            dict_parameters[parameter] = value_funcs[
                func.parameters[parameter].type
            ](llm, tokenized_context + llm.encode(str_dict)[0].tolist())
            print(dict_parameters)
        return dict_parameters
