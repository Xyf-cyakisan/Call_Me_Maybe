# from llm_sdk import Small_LLM_Model
# from .JSONParser import JSONParser
from argparse import ArgumentParser


def main() -> None:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--functions_definition", type=str,
                            default="./data/input/functions_definition.json")

    arg_parser.add_argument("--input", type=str,
                            default="./data/input/function_calling_tests.json")

    arg_parser.add_argument("--output", type=str,
                            default="./data/output/function_calls.json")
    arguments = arg_parser.parse_args()
    print(arguments)


if __name__ == "__main__":
    main()
