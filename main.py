#!/usr/bin/env python3

import json
from pydantic import BaseModel, Field
from llm_sdk.llm_sdk import Small_LLM_Model


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
        print(func)

    model = Small_LLM_Model()
    tokens = model.encode("Hello my name is Di")
    print("\ntokens:", tokens)
    print("\ntokens.tolist():", tokens.tolist())

    # input_ids = model.decode(tokens)
    # print("\ninput_ids:", input_ids)
    
    logits = model.get_logits_from_input_ids(tokens.tolist()[0])
    print("\nlogits:", logits[:10])
    print("\nlogits max:", max(logits))

    path = model.get_path_to_vocab_file()
    print("\npath:", path)

    with open(path, "r", encoding="utf-8") as f:
        vocabulary = f.read()
    print("\nvocabulary:", vocabulary[:500])


if __name__ == "__main__":
    main()
