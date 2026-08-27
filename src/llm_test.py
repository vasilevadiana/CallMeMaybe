#!/usr/bin/env python3

import json
from pydantic import BaseModel, Field
from llm_sdk.llm_sdk import *
from enum import Enum
import numpy as np


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

    START = "expect_open_brace"

    # "prompt"
    EXPECT_PROMPT_KEY_OPEN_QUOTE = "\""
    EXPECT_PROMPT_KEY = "prompt"
    EXPECT_PROMPT_KEY_CLOSE_QUOTE = "\""
    EXPECT_PROMPT_COLON = ":"
    EXPECT_PROMPT_SPACE = " "
    EXPECT_PROMPT_VALUE_OPEN_QUOTE = "\""
    EXPECT_PROMPT_VALUE = None
    EXPECT_PROMPT_VALUE_CLOSE_QUOTE = "\""
    EXPECT_PROMPT_COMMA = ","
    

    # "name"
    EXPECT_NAME_KEY = "expect_name_key"
    EXPECT_NAME_COLON = "expect_name_colon"
    EXPECT_NAME_VALUE = "expect_name_value"

    # "parameters"
    EXPECT_PARAMETERS_KEY = "expect_parameters_key"
    EXPECT_PARAMETERS_COLON = "expect_parameters_colon"
    EXPECT_PARAMETERS_VALUE = "expect_parameters_value"

    # Finished
    END = "expect_close_brace"


    # OPEN_BRACET = "{"
    # CLOSE_BRACET = "}"
    # QUOTE = "\""
    # COLON = ":"
    # COMMA = ","
    # FUNCTION_NAME = ""
    # PARAMETER_NAME = ""
    # PARAMETER_TYPE = ""


class ConstraintState(BaseModel):
    state: DecoderStates = DecoderStates.START
    prompt: str = ""
    function_name: str = ""
    generated_text: str = ""

    # def next(self) -> DecoderStates:
    #     """Change state to the next in enum declaration order."""
    #     states = list(DecoderStates)
    #     current_index = states.index(self.state)

    #     if current_index < len(states) - 1:
    #         self.state = states[current_index + 1]

    #     return self.state

            

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

    # if state == DecoderStates.COMPLETE:
    #     token_id = vocabulary["}"]

    #     if token_id is None:
    #         raise ValueError("Vocabulary does not contain '}'.")

    #     return {token_id}

    return set()


def mask_logits(logits: list[int], allowed_token_ids: set[int]) -> list[int]:
    """Mask all tokens that are not allowed."""

    masked_logits = [logits[i] if i in allowed_token_ids else float("-inf") for i in range(len(logits))]

    return masked_logits


def matching_functions(prefix: str, functions: list[FunctionDefinition]) -> list[FunctionDefinition]:

    return [function for function in functions if function.name.startswith(prefix)]


def is_valid_prefix(prefix: str, function_names: list[str]) -> bool:
    """
    Return True if prefix can still become one of the
    available function names.
    """
    # print("function_names:", function_names)
    # print(function_names[0].name.startswith(prefix))
    return any(function_name.name.startswith(prefix)for function_name in function_names)


def is_complete_function_name(prefix: str, functions: list[FunctionDefinition]) -> bool:
    """
    Return True if prefix is exactly one of the
    available function names.
    """

    return prefix in [func.name for func in functions]


def get_allowed_function_tokens(prefix: str, function_names: list[str], vocabulary: json) -> set[int]:
    """
    Find all vocabulary tokens that can be appended to prefix
    while keeping the result a valid prefix of at least one
    function name.

    Example:

        functions:
            fn_greet
            fn_add_numbers
            fn_reverse_string

        prefix:
            fn_

        If vocabulary contains:
            "g"
            "a"
            "r"
            "x"

        then:
            g -> allowed
            a -> allowed
            r -> allowed
            x -> forbidden
    """

    allowed_token_ids: set[int] = set()

    for token_text, token_id in vocabulary.items():
        # print(token_id, token_text)
        candidate = prefix + token_text

        if is_valid_prefix(candidate, function_names):
            allowed_token_ids.add(token_id)

    return allowed_token_ids


def get_allowed_number_tokens(vocabulary: json) -> set[int]:
    """
    Find all vocabulary tokens that are numbers (0-9).
    """

    allowed_token_ids: set[int] = set()

    for token_text, token_id in vocabulary.items():
        if token_text.isdigit(): ### TODO handle negative parameters and decimal numbers
            allowed_token_ids.add(token_id)

    return allowed_token_ids


def get_function_name(model, prompt, functions, vocabulary) -> FunctionDefinition:
    
    # logits = model.get_logits_from_input_ids(input_ids)
    # allowed = get_allowed_function_tokens("fn_", functions, vocabulary)
    # masked = mask_logits(logits, allowed)
    # next_token_id = masked.index(max(masked))
    # print("allowed function tokens:", allowed)
    # print("allowed function tokens decoded:", [model.decode(token_id) for token_id in allowed])
    # print("function names:", [func.name for func in functions])
    # print("selected token id:", next_token_id)
    # print("selected token:", repr(model.decode(next_token_id)))

    max_func_name = max([len(func.name) for func in functions])
    func_name = "fn_" #TODO not always starts with fn_
    input_ids = model.encode(prompt).tolist()[0]
    for i in range(max_func_name):
        logits = model.get_logits_from_input_ids(input_ids)
        allowed = get_allowed_function_tokens(func_name, functions, vocabulary)
        masked = mask_logits(logits, allowed)
        next_token_id = int(np.argmax(masked))
        input_ids.append(next_token_id)
        func_name += model.decode(next_token_id)
        if is_complete_function_name(func_name, functions):
            break
    
    for func_def in functions:
        if func_def.name == func_name:
            function_defenition = func_def
    print(function_defenition.name, function_defenition.parameters, function_defenition.description, function_defenition.returns)
    return (function_defenition)


def test_llm():
    print("############################################## FUNC_NAME TEST #################################################")
    functions = get_function_defenitions("data/input/functions_definition.json")
    
    for func in functions:
        print("\nfunction defenitions: ", func)

    model = Small_LLM_Model()
    prompt = "What is the sum of 2 and 4?"

    path_vocab = model.get_path_to_vocab_file()
    with open(path_vocab, "r", encoding="utf-8") as f:
        vocabulary = json.loads(f.read())

    function_defenition = get_function_name(model, prompt, functions, vocabulary)
    print(function_defenition)

    print("############################################## PARAMETERS TEST #################################################")

    input_ids = model.encode(prompt).tolist()[0]
    logits = model.get_logits_from_input_ids(input_ids)
    allowed = get_allowed_number_tokens(vocabulary)
    masked = mask_logits(logits, allowed)
    next_token_id = masked.index(max(masked))
    input_ids.append(next_token_id)
    print("selected token id:", next_token_id)
    print("selected token:", repr(model.decode(next_token_id)))

    logits = model.get_logits_from_input_ids(input_ids)
    allowed = get_allowed_number_tokens(vocabulary)
    masked = mask_logits(logits, allowed)
    next_token_id = masked.index(max(masked))
    input_ids.append(next_token_id)
    print("selected token id:", next_token_id)
    print("selected token:", repr(model.decode(next_token_id)))


    

    