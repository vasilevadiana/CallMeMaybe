#!/usr/bin/env python3

import json
from pydantic import BaseModel, Field
from llm_sdk.llm_sdk import *
from enum import Enum


class FunctionCalling(BaseModel):
    prompt: str


class ParameterDefinition(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: dict[str, str]


class DecoderStates(str, Enum):
    START = "START"
    OBJECT_OPEN = "OBJECT_OPEN"
    NAME_KEY = "NAME_KEY"
    NAME_COLON = "NAME_COLON"
    NAME_VALUE = "NAME_VALUE"
    COMPLETE = "COMPLETE"


class ConstraintState:
    state: DecoderStates = DecoderStates.START


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


def allowed_tokens(state: ConstraintState, vocabulary: json) -> set[int]:
    """Return token IDs allowed by the current structural state."""

    if state.state == DecoderStates.START:
        token_id = vocabulary["{"]

        if token_id is None:
            raise ValueError("Vocabulary does not contain '{'.")

        return {token_id}

    if state.state == DecoderStates.COMPLETE:
        token_id = vocabulary["}"]

        if token_id is None:
            raise ValueError("Vocabulary does not contain '}'.")

        return {token_id}

    return set()


def mask_logits(logits: list[int], allowed_token_ids: set[int]) -> list[int]:
    """Mask all tokens that are not allowed."""

    masked_logits = [float("-inf") * len(logits)]

    for token_id in allowed_token_ids:
        masked_logits[token_id] = logits[token_id]

    return masked_logits


def test_llm():
    functions = get_function_defenitions("data/input/functions_definition.json")
    
    # OUTPUT JSON:
    # [
    #     {
    #     "prompt": "What is the sum of 2 and 3?",
    #     "name": "fn_add_numbers",
    #     "parameters": {"a": 2.0, "b": 3.0}
    #     },
    #     {
    #     "prompt": "Reverse the string 'hello'",
    #     "name": "fn_reverse_string",
    #     "parameters": {"s": "hello"}
    #     }
    # ]

    for func in functions:
        print("\nfunction defenitions: ", func)

    model = Small_LLM_Model()
    prompt = "What is the sum of 2 and 4?"
    input_ids = model.encode(prompt).tolist()[0]

    print("===============================================")
    for i in range(11):
        logits = model.get_logits_from_input_ids(input_ids)
        
        # allowed_token_ids = constraint.get_allowed_tokens(...)
        # logits = mask_logits(logits, allowed_token_ids)

        next_token_id = logits.index(max(logits))
        print(next_token_id)
        input_ids.append(next_token_id)
        text = model.decode(next_token_id)
        print(repr(text))
        print("___")
    answer = model.decode(input_ids)
    print(answer)
    print("===============================================")

    for func in functions:
        print(func.name)
        input_ids = model.encode(func.name).tolist()[0]
        for id in input_ids:
            answer = model.decode(id)
            print(answer)
        print("_______________")
        


    path_vocab = model.get_path_to_vocab_file()
    print("\npath_vocab:", path_vocab)

    with open(path_vocab, "r", encoding="utf-8") as f:
        vocabulary = json.loads(f.read())
    print("\nvocabulary len:", len(vocabulary))
    # print(str(vocabulary)[:50])
    print("___________________________")
    print(vocabulary["Hello"]) #9707
    print(vocabulary["!"]) #0
    print(model.decode(9707))
    print(model.decode(0))


    #########################
    state = ConstraintState()

    allowed = allowed_tokens(state, vocabulary)

    print("Allowed token IDs:", allowed)

    masked = mask_logits(logits, allowed)

    next_token_id = int(masked).index(max(masked))

    print("Selected token ID:", next_token_id)
    print("Selected token:", repr(vocabulary[next_token_id]))