from .CallMeMaybeApplication import CallMeMaybeApplication
import sys
import json
from argparse import ArgumentParser
try:
    from pydantic import ValidationError
    from colorama import Fore, Style
except ImportError:
    print("ERROR: all dependencies are not installed, please run 'uv sync'"
          " before starting the program.")
    sys.exit(7)


def main() -> None:
    try:
        print("\033c")
        arg_parser = ArgumentParser()
        arg_parser.add_argument("--functions_definition",
                                type=str,
                                default="./data/input/functions_definition."
                                "json")

        arg_parser.add_argument("--input", type=str,
                                default="./data/input/function_calling_tests."
                                "json")

        arg_parser.add_argument("--output", type=str,
                                default="./data/output/function_calls.json")
        arguments = arg_parser.parse_args()
        app = CallMeMaybeApplication(arguments=arguments)
        app.run()
    except FileNotFoundError as not_found_err:
        print(Fore.RED, f"ERROR: {not_found_err}", Style.RESET_ALL)
        sys.exit(1)
    except PermissionError as permission_err:
        print(Fore.RED, f"ERROR: {permission_err}", Style.RESET_ALL)
        sys.exit(2)
    except json.JSONDecodeError as json_err:
        print(Fore.RED, f"ERROR: {json_err}", Style.RESET_ALL)
        sys.exit(3)
    except ValidationError as valid_errs:
        for err in valid_errs.errors():
            print(Fore.RED, f"ERROR: {repr(err['input'])} is not valid",
                  Style.RESET_ALL)
        sys.exit(4)
    except ValueError as val_err:
        print(Fore.RED, f"ERROR: {val_err}", Style.RESET_ALL)
        sys.exit(5)
    except KeyboardInterrupt:
        print(Fore.BLUE, "Bye-bye", Style.RESET_ALL)
        sys.exit(6)
    except Exception as err:
        print(Fore.RED + f"Unexpected ERROR: {err}" + Style.RESET_ALL)
        sys.exit(8)


if __name__ == "__main__":
    main()
