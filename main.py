#!/usr/bin/env python3

import json
from pydantic import BaseModel, Field
from llm_sdk.llm_sdk import *


class FunctionCalling(BaseModel):
    prompt: str


class ParameterDefinition(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: dict[str, str]


def get_function_defenitions(path: str) -> list[FunctionDefinition]:
    """Load function definitions from JSON file and validate"""
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        functions = [FunctionDefinition.model_validate(item) for item in data]
    except FileNotFoundError:
        raise ValueError(f"Function definition file not found: {path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in function definition file: {path}")

    return functions


def main():
    functions = get_function_defenitions("data/input/functions_definition.json")

    for func in functions:
        print("\nfunction defenitions: ", func)

    model = Small_LLM_Model()
    prompt = "What is the sum of 2 and 4?"
    input_ids = model.encode(prompt).tolist()[0]

    print("===============================================")
    for i in range(11):
        logits = model.get_logits_from_input_ids(input_ids)
        next_token_id = logits.index(max(logits))
        print(next_token_id)
        input_ids.append(next_token_id)
        text = model.decode(next_token_id)
        print(text)
        print("___")
    answer = model.decode(input_ids)
    print(answer)
    print("===============================================")


    # path_vocab = model.get_path_to_vocab_file()
    # print("\npath_vocab:", path_vocab)

    # with open(path_vocab, "r", encoding="utf-8") as f:
    #     vocabulary = json.loads(f.read())
    # print("\nvocabulary len:", len(vocabulary))

    # print("___________________________")
    # print(vocabulary["Hello"])
    # print(vocabulary["!"])


if __name__ == "__main__":
    main()
