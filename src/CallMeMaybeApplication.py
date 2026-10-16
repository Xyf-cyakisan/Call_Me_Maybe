from argparse import Namespace
from pydantic import BaseModel
from .JSONParser import JSONParser


class CallMeMaybeApplication(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    arguments: Namespace

    def __create_parser(self) -> None:
        self.__parser: JSONParser = JSONParser(
            func_definition_file_name=self.arguments.functions_definition,
            calling_tests_file_name=self.arguments.input,
            calls_json_file_name=self.arguments.output)

    def run(self) -> None:
        self.__create_parser()
        z, c = self.__parser.parse()
