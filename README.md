*This project has been created as part of the 42 curriculum by cyakisan.*

# Call Me Maybe

## Description

Call Me Maybe is a Python implementation of function calling for a small language
model. It converts natural-language prompts into structured function calls instead
of directly answering the prompts.

For example, a prompt such as `What is the sum of 2 and 3?` is converted into a
call to `fn_add_numbers` with typed arguments:

```json
{
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {
        "a": 2.0,
        "b": 3.0
    }
}
```

The project uses the `Qwen/Qwen3-0.6B` model through the local `llm_sdk`
package. Constrained decoding is used both to select a function and to generate
valid, schema-compliant JSON arguments.

## Features

- Reads function definitions and prompts from JSON arrays.
- Selects the function with the language model rather than hardcoded heuristics.
- Supports `string`, `number`, `integer`, and `boolean` parameters.
- Validates input models and generated values with Pydantic.
- Rejects duplicate JSON keys, unknown fields, missing parameters, and invalid
  parameter types.
- Produces one JSON output file containing the generated calls.
- Reports malformed input, missing files, permissions errors, validation errors,
  and generation failures with explicit messages.

## Project structure

```text
.
├── data/
│   └── input/
│       ├── function_calling_tests.json
│       └── functions_definition.json
├── llm_sdk/
│   └── llm_sdk/
├── src/
│   ├── CallMeMaybeApplication.py
│   ├── FunctionSelector.py
│   ├── JSONParser.py
│   ├── LLMOrchestrator.py
│   ├── parameters_models.py
│   └── __main__.py
├── Makefile
├── pyproject.toml
└── uv.lock
```

## Instructions

### Requirements

- Python 3.10 or later
- `uv`
- Internet access the first time the Hugging Face model is downloaded

### Installation

Install the project and its dependencies from the repository root:

```bash
make install
```

This runs `uv sync` and creates the project virtual environment.

### Running the program

Run with the default input and output paths:

```bash
make run
```

The equivalent command is:

```bash
uv run python -m src
```

By default:

- function definitions are read from
  `data/input/functions_definition.json`;
- prompts are read from `data/input/function_calling_tests.json`;
- calls are written to `data/output/function_calls.json`.

Custom files can be supplied with the command-line options:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

The output directory is created automatically. The output file must have a
`.json` extension.

## Example usage

After installation, run the sample prompts with:

```bash
uv run python -m src
```

The program creates `data/output/function_calls.json`. For example, the prompt
`What is the sum of 2 and 3?` produces a call with this structure:

```json
{
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {
        "a": 2.0,
        "b": 3.0
    }
}
```

To use another set of definitions and prompts:

```bash
uv run python -m src \
  --functions_definition path/to/functions.json \
  --input path/to/prompts.json \
  --output path/to/function_calls.json
```

### Debugging and linting

```bash
make debug
make lint
make lint-strict
make clean
```

`make debug` starts Python's built-in debugger. `make lint` runs Flake8 and the
required Mypy checks; `make lint-strict` additionally runs Mypy in strict mode.

## Input format

### Function definitions

`functions_definition.json` must contain an array of function objects. Each
function has a unique non-empty `name`, a `description`, a `parameters` object,
and a `returns` type:

```json
[
    {
        "name": "fn_add_numbers",
        "description": "Add two numbers together and return their sum.",
        "parameters": {
            "a": {"type": "number"},
            "b": {"type": "number"}
        },
        "returns": {"type": "number"}
    }
]
```

Supported types are `string`, `number`, `integer`, and `boolean`. Unknown fields
are rejected.

### Prompts

`function_calling_tests.json` must contain an array of objects with one
non-empty `prompt` field:

```json
[
    {"prompt": "What is the sum of 2 and 3?"},
    {"prompt": "Reverse the string 'hello'"}
]
```

## Algorithm

The generation pipeline has two constrained-decoding stages.

### 1. Function selection

For each prompt, the application builds a context containing every function
name and description. `FunctionSelector` tokenizes each valid function name and
generates the answer one token at a time.

At every step, only tokens that continue at least one valid function name are
allowed. If several functions share the current prefix, the model logits choose
the highest-scoring allowed token. Tokens outside the allowed set are masked
with negative infinity. Generation stops only when the token sequence exactly
matches one of the declared function names.

This guarantees that the selected name belongs to the input definitions while
still letting the LLM make the semantic choice.

### 2. Argument generation

For the selected function, `LLMOrchestrator` builds a tool schema containing the
parameter names and their types. The model is prompted to produce only the
arguments object.

`ParametersGenerator` then performs token-by-token constrained decoding:

1. `JsonPrefixChecker` parses the decoded prefix as an incremental JSON object.
2. The checker accepts only valid JSON syntax and the next key/value expected by
   the function schema.
3. String, boolean, integer, and number values each have dedicated validation
   rules.
4. Candidate tokens are ordered by model logits, and candidates that would make
   the prefix invalid are discarded.
5. Generation ends only when every required parameter is present and the JSON
   object is complete.
6. The completed object is parsed and validated again before it is returned.

The generator has a limit of 512 generated tokens. If no token can continue the
schema or the limit is reached, it raises an explicit error instead of writing
an invalid result.

## Design decisions

- **Pydantic models:** Input definitions, prompts, and output calls use Pydantic
  models so malformed data is rejected at the boundary.
- **Local SDK:** `llm_sdk` provides the model and the public operations needed
  by the project (`encode`, `decode`, logits, and vocabulary access) without
  coupling the application to model internals.
- **Two separate constraints:** Function names are best handled as a finite
  set of valid token sequences, while arguments require a JSON-prefix state
  machine. Keeping these concerns separate makes both stages easier to verify.
- **Schema-driven generation:** The decoder derives parameter names and types
  from the input definitions, so the implementation is not hardcoded to the
  sample prompts.
- **Explicit error handling:** Parsing and execution errors are converted into
  clear command-line messages with distinct exit codes.
- **Vocabulary caching:** Vocabulary token IDs are cached by vocabulary path to
  avoid repeatedly reading the same model vocabulary for multiple prompts.

## Performance and reliability

The main cost is model inference: each generated token requires a logits
request, and the model is loaded once for the complete run. Function selection
may also compare logits against the allowed function-name tokens. Argument
generation is capped at 512 tokens per function call.

The constrained decoder is designed to provide 100% valid JSON and
schema-compliant arguments for successful generations, independently of whether
the small model would have produced valid JSON by prompting alone. Semantic
quality, such as selecting the intended function and extracting the intended
values, still depends on the model and the wording of the prompt. It should be
measured with representative inputs rather than assumed from structural
validity.

The model is instantiated once in `LLMOrchestrator`, which avoids reloading it
for every prompt. The implementation processes prompts sequentially, keeping
memory usage predictable and preserving the order of the input prompts.

## Challenges faced

- **Small-model reliability:** A small model can produce incomplete or malformed
  JSON. The solution was to constrain every candidate token and validate the
  completed object.
- **Prefix ambiguity:** Several function names can share token prefixes. The
  selector keeps all compatible prefixes until the model can distinguish them.
- **Typed JSON values:** JSON strings, booleans, integers, and floating-point
  numbers have different lexical rules. `JsonPrefixChecker` handles these
  states incrementally instead of validating only after generation.
- **Literal text and regular expressions:** Prompts can contain punctuation,
  quotes, whitespace, and regex patterns. The argument prompt explicitly asks
  the model to preserve literal text, while JSON string parsing handles the
  required escaping.
- **Input changes during review:** Definitions and prompts are read at runtime
  and validated dynamically; the examples are not hardcoded into the algorithm.

## Testing strategy

The supplied JSON files provide a representative smoke test covering arithmetic,
greetings, string reversal, square roots, and substitutions with multiple
parameters.

For a complete check:

1. Run the program with the default files.
2. Parse the generated file with a JSON parser.
3. Check that the number and order of calls match the input prompts.
4. Verify that every function name exists in `functions_definition.json`.
5. Verify that the parameter names and JSON types match the selected definition.
6. Try malformed JSON, missing files, duplicate keys, empty definitions,
   unknown fields, empty strings, special characters, and functions with
   multiple parameters.
7. Run `make lint` and, when possible, `make lint-strict`.

The output directory is generated at runtime and should not be committed.

## Resources

- [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers/)
  — model and tokenizer concepts.
- [Qwen model page](https://huggingface.co/Qwen/Qwen3-0.6B) — the default model
  used by the local SDK.
- [Python `json` documentation](https://docs.python.org/3/library/json.html) —
  JSON parsing and serialization.
- [Pydantic documentation](https://docs.pydantic.dev/) — runtime data
  validation and typed models.
- [Mypy documentation](https://mypy.readthedocs.io/) — static type checking.
- [Flake8 documentation](https://flake8.pycqa.org/) — Python style and linting.

### Use of AI

AI assistance was used to inspect the repository, compare the implementation
with the assignment requirements, and draft and structure this README. It was
not used as a substitute for understanding the decoding algorithm or validating
the project's behavior; the documented commands, file formats, and design
decisions were checked against the source files in this repository.
