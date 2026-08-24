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
    START = "start"

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

    # Top-level object
    EXPECT_OPEN_BRACE = "expect_open_brace"
    EXPECT_TOP_LEVEL_KEY = "expect_top_level_key"
    EXPECT_TOP_LEVEL_COLON = "expect_top_level_colon"

    # "prompt"
    EXPECT_PROMPT_VALUE = "expect_prompt_value"

    # "name"
    EXPECT_NAME_VALUE = "expect_name_value"

    # "parameters"
    EXPECT_PARAMETERS_VALUE = "expect_parameters_value"

    EXPECT_PARAMETER_KEY = "expect_parameter_key"
    EXPECT_PARAMETER_COLON = "expect_parameter_colon"
    EXPECT_PARAMETER_VALUE = "expect_parameter_value"

    # Finished
    EXPECT_CLOSE_PARAMETERS = "expect_close_parameters"
    EXPECT_CLOSE_OBJECT = "expect_close_object"

    COMPLETE = "complete"


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

    if state == DecoderStates.COMPLETE:
        token_id = vocabulary["}"]

        if token_id is None:
            raise ValueError("Vocabulary does not contain '}'.")

        return {token_id}

    return set()


def mask_logits(logits: list[int], allowed_token_ids: set[int]) -> list[int]:
    """Mask all tokens that are not allowed."""

    masked_logits = [logits[i] if i in allowed_token_ids else float("-inf") for i in range(len(logits))]

    return masked_logits


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
    state = state.next()
    print(state == DecoderStates.EXPECT_OPEN_BRACE)

    allowed = allowed_tokens(state, vocabulary)
    masked = mask_logits(logits, allowed)
    next_token_id = masked.index(max(masked))

    print("allowed token ids:", allowed)
    print("selected token id:", next_token_id)
    print("selected token:", repr(model.decode(next_token_id)))

    state = state.next()
    print(state == DecoderStates.EXPECT_OPEN_BRACE)
    