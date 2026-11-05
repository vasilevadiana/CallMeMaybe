*This project has been created as part of the 42 curriculum by dvasilev.*

# Call Me Maybe

## Description

Call Me Maybe converts a natural-language request into a structured function
call. Given a set of function definitions and a prompt such as “What is the sum
of 2 and 3?”, it selects a defined function and extracts its arguments; it does
not execute the selected function or calculate its return value.

The program uses the `Qwen/Qwen3-0.6B` causal language model through the
provided `llm_sdk` package. Rather than asking the model to generate a complete
JSON object freely, the implementation restricts each next-token choice to
tokens that fit the current function name or parameter value.

## Features

- Validates input file structure and reports invalid or unreadable input files.
- Selects a function only from the supplied definitions.
- Extracts `string`, `number`, `integer`, and `boolean` parameters.
- Writes one result per input prompt as a JSON array.


## Requirements

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/)
- Network access on run to download the model and tokenizer from Hugging
  Face; sufficient disk space for the Python dependencies and model files

## Instructions

From the repository root, install the project dependencies:

```bash
make install
```

Run with the included input files and default output path:

```bash
make run
```

By default, the program reads:

- `data/input/functions_definition.json`
- `data/input/function_calling_tests.json`

and writes `data/output/function_calling_results.json`. The output directory is
created if needed and is git-ignored.

To supply different files, use the command-line options:

```bash
uv run python -m src \
  --functions_definition path/to/function_definitions.json \
  --input path/to/prompts.json \
  --output path/to/results.json
```
But if there are no enought space in disk transfer all denendencies to goinfre folder:
```bash
mkdir -p "/goinfre/$USER/.venvs" "/goinfre/$USER/tmp" "/goinfre/$USER/callmemaybe-output"
UV_PROJECT_ENVIRONMENT="/goinfre/$USER/.venvs/callmemaybe" \
TMPDIR="/goinfre/$USER/tmp" \
HF_HOME="/goinfre/$USER/.cache/huggingface" \
uv run python -m src \
  --functions_definition path/to/function_definitions.json \
  --input path/to/prompts.json \
  --output path/to/results.json
```

Both input files must contain JSON arrays. Each prompt entry has a `prompt`
string:

```json
[
  {"prompt": "What is the sum of 2 and 3?"}
]
```

Each function definition has a `name`, `description`, `parameters`, and
`returns`. Parameter definitions specify a supported `type`:

```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together.",
    "parameters": {
      "a": {"type": "number"},
      "b": {"type": "number"}
    },
    "returns": {"type": "number"}
  }
]
```

The result file contains an array of objects with exactly `prompt`, `name`, and
`parameters` fields. For example:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  }
]
```

## Example usage

Run the bundled examples and explicitly set all three paths:

```bash
make run
```

For the prompt “Greet shrek”, an expected result has this shape:

```json
{
  "prompt": "Greet shrek",
  "name": "fn_greet",
  "parameters": {"name": "shrek"}
}
```

## Algorithm: constrained decoding

The generation is performed token by token. At each step, the model returns
logits for the next token. The decoder builds an allowed set for the current
part of the function call, masks all other token scores with negative infinity,
and chooses the allowed token with the highest logit.

1. **Choose a function.** A prompt containing the user request and available
   function descriptions is sent to the model. The only allowed next tokens
   are those whose decoded text continues a prefix of one of the defined
   function names. Generation stops when a complete name and its closing quote
   have been produced.
2. **Generate parameters in definition order.** The JSON key for each
   supported parameter is added to the context before its value is generated.
   The maximum number of generated tokens for each value is the prompt's
   character length.
3. **Constrain values by type.**
   - For a **number** or **integer**, candidate tokens must leave the value as
     a valid numeric prefix. A comma, closing brace, or newline can end a
     value only after it is complete. Integers must contain digits and an
     optional leading minus; numbers may also contain a decimal point.
   - For a **string**, candidate tokens may not contain a quote or newline.
     Quote and JSON-boundary tokens are allowed as terminators.
   - For a **boolean**, only vocabulary tokens decoding exactly to `true` or
     `false` can form the value; comma and closing-brace tokens can terminate
     it.

This approach constrains function names and value tokens instead of relying on
the model to produce a complete JSON document correctly. JSON serialization
ensures the output file is syntactically valid JSON.

## Design decisions

- **Use the supplied small local model.** The SDK wraps Qwen3-0.6B and selects
  an available accelerator or CPU.
- **Use greedy selection among valid tokens.** Choosing the highest-logit
  allowed token avoids sampling variability while preserving the model's
  preference among structurally acceptable choices.
- **Process parameters in definition order.** The model sees each parameter
  key before generating its value, keeping the prompt aligned with the
  declared function schema.

## Performance and reliability

The project subject sets targets of at least 90% correct function/argument
extraction and processing the test prompts in under five minutes on standard
hardware. This repository includes 11 example prompts across five function
definitions.

Inference time depends on hardware, prompt length.

Constrained choices prevent invalid function names and reject malformed
numeric and boolean values. However, constrained decoding does not guarantee
that the model understood the request correctly. It also does not currently
support quoted or multiline string values, unknown parameter types, or
specialized schema rules such as numeric ranges. Invalid or unavailable token
choices raise an error rather than falling back to unconstrained generation.

## Testing strategy

Run the program:

```bash
make run
```

Check that the result file:

```bash
data/output/function_calling_results.json
```

For functional review, compare each result with its input prompt and function
definition: confirm that every prompt has one result, the name is defined, all
supported parameters are present, and their JSON types match the definition.

## Challenges faced

Challenge was to balance constraints with value generation.
The decoder therefore uses separate candidate rules and terminators for
function names, numbers, strings, and booleans. This keeps the supported values
within predictable boundaries, while the current string-token restrictions
make quotes and newlines a known limitation.

## Resources

- [Hugging Face Hub documentation](https://huggingface.co/docs/huggingface_hub/)
  — downloading model files.
- [A Guide to Structured Generation Using Constrained Decoding](https://www.aidancooper.co.uk/constrained-decoding/)
- [Controlling your LLM: Deep dive into Constrained Generation](https://medium.com/@docherty/controlling-your-llm-deep-dive-into-constrained-generation-1e561c736a20)

**AI usage:** AI was used to understand the subject, evaluate implementation ideas, review docstrings, troubleshoot mypy and flake8 errors, and draft this README.
