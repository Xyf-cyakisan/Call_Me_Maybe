from typing import Any, Generator
from pydantic import BaseModel
from .FunctionSelector import FunctionSelector
from .parameters_models import ParametersGenerator
from .output_model import FunctionCall
from .input_models import Function, Prompt
from llm_sdk import Small_LLM_Model
import json


class LLMOrchestrator(BaseModel):
    functions: list[Function]
    prompts: list[Prompt]

    def __build_funcs_name_contexts(self) -> Generator[str]:
        for prompt in self.prompts:
            context = ""
            for i, func in enumerate(self.functions, start=1):
                context += f"Function {i}: {func.name}\n"
                context += f"Description {i}: {func.description}\n\n"
            context += f"Prompt: {prompt.prompt}\n\n"
            context += ("Select the most appropriate function from the"
                        "list above.\n")
            context += ("Return the function name without any parameters.\n")
            context += "Function: "
            yield context

    @staticmethod
    def tool_definition(
        function: Function,
        name: str | None = None,
        description: str | None = None,
        parameters: list[str] | None = None,
    ) -> dict[str, object]:
        """Build a JSON-schema-style tool definition."""
        parameter_names = (
            list(function.parameters)
            if parameters is None
            else parameters
        )
        properties = {
            parameter: {
                "type": function.parameters[parameter].type
            }
            for parameter in parameter_names
        }

        return {
            "type": "function",
            "function": {
                "name": (
                    function.name
                    if name is None
                    else name
                ),
                "description": (
                    function.description
                    if description is None
                    else description
                ),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": parameter_names,
                },
            },
        }

    def build_argument_context(
        self,
        prompt: Prompt,
        function: Function,
        parameters: list[str],
    ) -> str:
        """Build one shared context for typed argument generation."""
        tool = self.tool_definition(
            function,
            parameters=parameters,
        )

        return (
            "<|im_start|>system\n"
            "You are a function-calling assistant. Supply only the "
            "arguments described in the provided tool schema. "
            "Use the JSON types required by the schema. "
            "Do not execute the function or return its result.\n"
            "Copy literal text without changing its spelling, case, "
            "or whitespace. Translate symbol names into literal "
            "characters when appropriate. For a requested class of text, "
            "use a pattern matching that class rather than only one "
            "example.\n"
            "Arguments already generated: "
            "{}\n\n"
            "# Tools\n\n"
            "Available function definitions:\n<tools>\n"
            f"{json.dumps(tool, ensure_ascii=False)}\n"
            "</tools>\n\n"
            "Write exactly one JSON function call between <tool_call> "
            "and </tool_call>, with the keys name and arguments.\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{prompt.prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n"
            "<tool_call>\n"
            '{"name": '
            f'{json.dumps(function.name)}, '
            '"arguments": '
        )

    def __set_llm(self) -> None:
        self.__llm = Small_LLM_Model()

    def __set_function_selector(self) -> None:
        self.__function_selector = FunctionSelector()

    def __set_parameters_generator(self) -> None:
        self.__parameters_generator = ParametersGenerator()

    def get_llm(self) -> Small_LLM_Model:
        return self.__llm

    def __create_functions_calls_objects(self, funcs_name: list[str],
                                         prompts: list[str],
                                         parameters: list[dict[str, Any]]
                                         ) -> list[FunctionCall]:
        return [
            FunctionCall(prompt=prompts[index],
                         name=funcs_name[index],
                         parameters=parameters[index])
            for index in range(len(funcs_name))
        ]

    def run_llm(self) -> list[FunctionCall]:
        self.__set_llm()
        self.__set_function_selector()
        self.__set_parameters_generator()
        functions_calls_data = {
            "funcs_name": [],
            "prompts": [],
            "parameters": []
        }
        needed_funcs = []
        for func_name_context in self.__build_funcs_name_contexts():
            functions_calls_data["funcs_name"].append(
                self.__function_selector.get_func_name(
                    self.get_llm(), func_name_context, self.functions))
            for func in self.functions:
                if functions_calls_data["funcs_name"][-1] == func.name:
                    needed_funcs.append(func)
                    break
        for func, prompt in zip(needed_funcs, self.prompts):
            parameters = list(func.parameters.keys())
            tokenized_context = self.__llm.encode(
             self.build_argument_context(
                prompt,
                func,
                parameters))[0].tolist()
            result = self.__parameters_generator.generate(
                func, prompt.prompt, self.get_llm(), tokenized_context)
            functions_calls_data["parameters"].append(result)
        for prompt in self.prompts:
            functions_calls_data["prompts"].append(prompt.prompt)
        return (self.__create_functions_calls_objects(**functions_calls_data))
