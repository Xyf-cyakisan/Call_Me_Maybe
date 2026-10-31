from pydantic import BaseModel
import json
from typing import Any
from .input_models import Prompt
from .input_models import Function


class JSONParser(BaseModel):
    func_definition_file_name: str
    calling_tests_file_name: str
    calls_json_file_name: str

    def __read_json_file(self, file_name: str) -> Any:
        try:
            with open(file_name, encoding="utf-8") as json_file:
                loaded = json.load(
                                 json_file,
                                 object_pairs_hook=(
                                     self.__check_duplicate_parameters
                                     )
                )
                if not isinstance(loaded, list):
                    raise ValueError(f"'{file_name}' is not a JSON array")
                return loaded
        except UnicodeDecodeError:
            raise ValueError(
                f"Invalid UTF-8 file '{file_name}'"
            )
        except json.JSONDecodeError as json_err:
            raise json.JSONDecodeError(
                f"Invalid JSON '{file_name}'",
                json_err.doc,
                json_err.pos) from json_err
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{file_name}' not found")
        except PermissionError:
            raise PermissionError(f"Cannot access file '{file_name}' because "
                                  "of permissions")

    def __check_duplicate_parameters(self, parameters: list[tuple[str, object]
                                                            ]) -> dict[str,
                                                                       object]:
        dict_parameters: dict[str, object] = {}
        for key, value in parameters:
            if key in dict_parameters:
                raise ValueError(f"Duplicate JSON key: '{key}'")
            dict_parameters[key] = value
        return dict_parameters

    def __check_duplicate_func_name(self, funcs: list[Function]) -> None:
        if len([func.name for func in funcs]) != len(set([func.name
                                                          for func in
                                                          funcs])):
            raise ValueError(f"File {self.func_definition_file_name} has "
                             "multiple functions named the same.")

    def parse(
            self) -> tuple[list[Function], list[Prompt]]:
        func_def_file = self.__read_json_file(self.func_definition_file_name)
        prompt_def_file = self.__read_json_file(self.calling_tests_file_name)
        for index, func_def in enumerate(func_def_file):
            if not isinstance(func_def, dict):
                raise ValueError(
                    f"Function definition at index {index} must be a JSON"
                    " object"
                )

        for index, prompt in enumerate(prompt_def_file):
            if not isinstance(prompt, dict):
                raise ValueError(
                    f"Prompt at index {index} must be a JSON object"
                )
        objects = (
            [Function(**func_def)
             for func_def in func_def_file],
            [Prompt(**prompt) for prompt in prompt_def_file]
        )
        if not objects[0]:
            raise ValueError(f"'{self.func_definition_file_name}' is empty")
        self.__check_duplicate_func_name(objects[0])
        return objects
