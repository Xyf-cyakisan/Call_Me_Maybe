from pydantic import BaseModel
import json
from typing import Any
from .models import Prompt
from .models import Function


class JSONParser(BaseModel):
    func_definition_file_name: str
    calling_tests_file_name: str
    calls_json_file_name: str

    def __read_json_file(self, file_name: str) -> Any:
        try:
            with open(file_name, encoding="utf-8") as json_file:
                return json.load(json_file)
        except json.JSONDecodeError:
            raise json.JSONDecodeError(f"Invalid JSON '{file_name}'")
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{file_name}' not found")
        except PermissionError:
            raise PermissionError(f"Cannot access file '{file_name}' because "
                                  "of permissions")

    def parse(
            self) -> tuple[list[Prompt], list[Function]]:
        if not self.calls_json_file_name.lower().endswith(".json"):
            raise ValueError("Filename must have a .json extension")
        return (
            [Function(**func_def) for func_def in self.__read_json_file(
                self.func_definition_file_name)],
            [Prompt(**prompt) for prompt in self.__read_json_file(
                self.calling_tests_file_name)]
        )
