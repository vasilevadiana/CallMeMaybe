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


class ConstraintState:
    state: DecoderStates = DecoderStates.START
    prompt: str
    function_name: str
    generated_text: str = ""

    def next(self) -> DecoderStates:
        """Change state to the next in enum declaration order."""
        states = list(DecoderStates)
        current_index = states.index(self.state)

        if current_index < len(states) - 1:
            self.state = states[current_index + 1]

        return self.state

            

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

    if state == DecoderStates.START:
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


def test_llm():
    functions = get_function_defenitions("data/input/functions_definition.json")

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
    print(str(vocabulary)[:50])
    print("___________________________")
    print(vocabulary["Hello"]) #9707
    print(vocabulary["!"]) #0
    print(vocabulary["{"]) #90
    print(model.decode(9707))
    print(model.decode(0))
    print(model.decode(90))



    print("######################### TEST #################################")
    state = ConstraintState()
    # state = state.next()
    # print(state == DecoderStates.EXPECT_OPEN_BRACE)

    allowed = allowed_tokens(state, vocabulary)
    masked = mask_logits(logits, allowed)
    next_token_id = masked.index(max(masked))

    print("allowed token ids:", allowed)
    print("selected token id:", next_token_id)
    print("selected token:", repr(model.decode(next_token_id)))

    state = state.next()
    # print(state == DecoderStates.EXPECT_OPEN_BRACE)

    print("############################################## FUNC_NAME TEST #################################################")
    allowed = get_allowed_function_tokens("fn_", functions, vocabulary)
    print("allowed function tokens:", allowed)
    print("allowed function tokens decoded:", [model.decode(token_id) for token_id in allowed])
    print("function names:", [func.name for func in functions])
    masked = mask_logits(logits, allowed)
    next_token_id = masked.index(max(masked))
    print("selected token id:", next_token_id)
    print("selected token:", repr(model.decode(next_token_id)))

    max_func_name = max([len(func.name) for func in functions])
    func_name = "fn_"
    for i in range(max_func_name):
        allowed = get_allowed_function_tokens(func_name, functions, vocabulary)
        print("allowed function tokens:", allowed)
        print("allowed function tokens decoded:", [model.decode(token_id) for token_id in allowed])
        masked = mask_logits(logits, allowed)
        next_token_id = masked.index(max(masked))
        print("selected token id:", next_token_id)
        print("selected token:", repr(model.decode(next_token_id)))
        func_name += model.decode(next_token_id)
        print("prefix:", repr(func_name))
        if is_complete_function_name(func_name, functions):
            break
    
    for func_def in functions:
        if func_def.name == func_name:
            function_defenition = func_def
    print(function_defenition.name, function_defenition.parameters, function_defenition.description, function_defenition.returns)

    print("############################################## PARAMETERS TEST #################################################")
    def fill_parameters_values(function_defenition: FunctionDefinition, prompt: str) -> dict[str, str | int]:
        print("hi")
        print(function_defenition)
        parameters = function_defenition.parameters
        print(function_defenition.parameters)
        return (parameters)


    par = fill_parameters_values(function_defenition, prompt)
    print(par)

    

    