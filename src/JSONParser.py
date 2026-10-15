from pydantic import BaseModel, model_validator, ValidationError, Field
import json
from typing import Any

class JSONParser(BaseModel):
    function_definitions_json: Any
    function_calling_tests_json: Any
    function_calls_json_file_name: str
    

    @staticmethod
    def __read_json_files()

    @classmethod
    def parse(cls ,json_files_name: list[str]) ->"JSONParser":
        definitions_json, tests_json = cls.__read_json_files(json_files_name)