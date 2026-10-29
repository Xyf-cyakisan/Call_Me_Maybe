from pydantic import BaseModel
import json
from typing import Any
from .input_models import Prompt
from .input_models import Function
from pathlib import Path


class JSONParser(BaseModel):
    func_definition_file_name: str
    calling_tests_file_name: str
    calls_json_file_name: str

    def __read_json_file(self, file_name: str) -> Any:
        try:
            with open(file_name, encoding="utf-8") as json_file:
                return json.load(json_file)
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

    def __create_output_file(self) -> None:
        if not self.calls_json_file_name.lower().endswith(".json"):
            raise ValueError("Filename must have a .json extension")
        path = Path(self.calls_json_file_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("w") as output_f:
                output_f.write("")
        except PermissionError:
            raise PermissionError(f"Cannot access file '{path}' because "
                                  "of permissions")

    def parse(
            self) -> tuple[list[Prompt], list[Function]]:
        objects = (
            [Function(**func_def)
             for func_def in self.__read_json_file(
                self.func_definition_file_name)],
            [Prompt(**prompt) for prompt in self.__read_json_file(
                self.calling_tests_file_name)]
        )
        self.__create_output_file()
        return objects
