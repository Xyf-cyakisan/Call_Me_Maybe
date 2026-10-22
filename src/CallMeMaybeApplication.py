from argparse import Namespace
from pydantic import BaseModel
from .models import Function, Prompt
from .JSONParser import JSONParser
from .LLMOrchestrator import LLMOrchestrator
import json


class CallMeMaybeApplication(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    arguments: Namespace

    def __create_parser(self) -> None:
        self.__parser: JSONParser = JSONParser(
            func_definition_file_name=self.arguments.functions_definition,
            calling_tests_file_name=self.arguments.input,
            calls_json_file_name=self.arguments.output)

    def __create_orchestrator(self, functions: list[Function],
                              prompts: list[Prompt]) -> None:
        self.__orchestrator = LLMOrchestrator(functions=functions,
                                              prompts=prompts)

    def run(self) -> None:
        self.__create_parser()
        funcs, prompts = self.__parser.parse()
        self.__create_orchestrator(funcs, prompts)
        result = [call.model_dump(mode="json") for call
                  in self.__orchestrator.run_llm()]
        with open(self.__parser.calls_json_file_name, "w") as output_json:
            json.dump(result,
                      output_json, indent=4)
