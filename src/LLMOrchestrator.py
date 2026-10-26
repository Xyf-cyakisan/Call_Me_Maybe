from typing import Generator
from collections import defaultdict
from pydantic import BaseModel
from .parameters_decoders import BoolDecoder, NumberDecoder, StringDecoder
from .output_models import FunctionCall
from .models import Function, Prompt
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

    def __build_parameter_context(
        self,
        prompt: str,
        function: Function,
        parameter: str,
        values: dict[str, object],
    ) -> str:
        context = ""

        context += f"Prompt: {prompt}\n"
        context += f"Function: {function.name}\n"
        context += f"Description: {function.description}\n"

        context += "Function signature: "
        context += f"{function.name}("

        parameters = []

        for name, definition in function.parameters.items():
            parameters.append(f"{name}: {definition.type}")

        context += ", ".join(parameters)
        context += ")\n"

        context += "Arguments already generated:\n"

        for name, value in values.items():
            context += f"{name}={value}\n"

        context += f"{parameter}="

        return context

    def __set_llm(self) -> None:
        self.__llm = Small_LLM_Model()

    def __set_tokenized_funcs(self) -> str:
        self.__tokenized_funcs: dict[str, list[int]] = {
            func.name: self.__llm.encode(func.name + '\n')[0].tolist()
            for func in self.functions}

    def __set_parameters_decoders(self) -> None:
        self.__parameters_decoders = defaultdict(BoolDecoder)
        self.__parameters_decoders["string"] = StringDecoder
        self.__parameters_decoders["number"] = NumberDecoder

    def get_llm(self) -> Small_LLM_Model:
        return self.__llm

    def __get_func_name(self, context: str) -> str:
        tokenized_context = self.__llm.encode(context)[0].tolist()
        tokenized_answer = []
        while True:
            allowed_ids = []
            logits = self.__llm.get_logits_from_input_ids(
                    tokenized_context + tokenized_answer
            )
            for tokened_func in self.__tokenized_funcs.values():
                if (len(tokened_func) >= len(tokenized_answer)
                   and tokenized_answer == tokened_func[:len(tokenized_answer)
                                                        ]):
                    allowed_ids.append(tokened_func[len(tokenized_answer)])
            next_token_id = logits.index((max([logits[allowed_id]
                                              for allowed_id in
                                              allowed_ids])))
            tokenized_answer.append(next_token_id)
            if tokenized_answer in self.__tokenized_funcs.values():
                str_func = self.__llm.decode(tokenized_answer[:-1])
                for func in self.functions:
                    if func.name == str_func:
                        return func

    def __get_parameters(self, func: Function, prompt: str) -> str:
        tokenized_answer = []
        beginning = True
        end = False
        parameters = list(func.parameters.keys())
        values = {}
        index = 0
        while True:
            if index < len(parameters):
                tokenized_context = (self.__llm.encode(self.__build_parameter_context(prompt, func, parameters[index], values))[0].tolist())
                allowed_ids = []
                logits = self.__llm.get_logits_from_input_ids(
                        tokenized_context + tokenized_answer
                )
            if beginning:
                allowed_ids = [self.__llm.encode("{")]
                beginning = not beginning
                next_token_id = logits.index((max([logits[allowed_id]
                                              for allowed_id in
                                              allowed_ids])))
                tokenized_answer.append(next_token_id)
            elif end:
                allowed_ids = [self.__llm.encode("}")]
                next_token_id = logits.index((max([logits[allowed_id]
                                              for allowed_id in
                                              allowed_ids])))
                tokenized_answer.append(next_token_id)
                return (self.__llm.decode(tokenized_answer))
            else:
                result = self.__parameters_decoders[
                    func.parameters[parameters[index]].type
                    ].constrained_decode(
                        values,
                        parameters[index], func, prompt.prompt,
                        self.get_llm(), tokenized_context,
                        tokenized_answer
                )
                end, tokenized_answer, values = result
                index += 1

    def __create_functions_calls_objects(self, funcs_name: list[str],
                                         prompts: list[str],
                                         parameters: list[dict[str, str]]
                                         ) -> list[FunctionCall]:
        return [
            FunctionCall(prompt=prompts[index],
                         name=funcs_name[index],
                         parameters=json.loads(parameters[index]))
            for index in range(len(funcs_name))
        ]

    def run_llm(self) -> list[FunctionCall]:
        self.__set_llm()
        self.__set_tokenized_funcs()
        self.__set_parameters_decoders()
        functions_calls_data = {
            "funcs_name": [],
            "prompts": [],
            "parameters": []
        }
        for func_name_context in self.__build_funcs_name_contexts():
            functions_calls_data["funcs_name"].append(self.__get_func_name(
                func_name_context))
        funcs_name = [func.name for func in functions_calls_data["funcs_name"]]
        parameters_gen = zip(functions_calls_data["funcs_name"], self.prompts)
        for func, prompt in parameters_gen:
            result = self.__get_parameters(
                func, prompt)
            functions_calls_data["parameters"].append(result)
        for prompt in self.prompts:
            functions_calls_data["prompts"].append(prompt.prompt)
        functions_calls_data["funcs_name"] = funcs_name
        return (self.__create_functions_calls_objects(**functions_calls_data))
