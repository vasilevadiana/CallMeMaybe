#!/usr/bin/env python3

import json
from pydantic import BaseModel, Field
from llm_sdk.llm_sdk import *
from enum import Enum
import numpy as np
import re
from typing import Any


class FunctionCalling(BaseModel):
    prompt: str


class ParameterDefinition(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: dict[str, str]


class FunctionCallResult(BaseModel):
    """One output entry: exactly prompt / name / parameters."""
 
    prompt: str
    name: str
    parameters: dict[str, Any]
            

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


def get_function_calling_tests(path: str) -> list[FunctionCalling]:
    """Load function calling tests from JSON file and validate"""
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        promts = [FunctionCalling.model_validate(item) for item in data]
    except FileNotFoundError:
        raise ValueError(f"Function definition file not found: {path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in function definition file: {path}")

    return promts


def decode_vocabulary(model: Small_LLM_Model,
                      vocabulary: dict[str, int]) -> dict[int, str]:
    """Map each token id to its real text using the SDK's decode."""
    id_to_text: dict[int, str] = {}
    for token_id in vocabulary.values():
        text = model.decode([token_id])
        if text and "\ufffd" not in text:
            id_to_text[token_id] = text
    return id_to_text


def test_llm():
    print("############################################## FUNC_NAME TEST #################################################")
    functions = get_function_defenitions("data/input/functions_definition.json")
    prompts = get_function_calling_tests("data/input/function_calling_tests.json")

    model = Small_LLM_Model()

    with open(model.get_path_to_vocab_file(), encoding="utf-8") as f:
        vocabulary = json.load(f)

    id_to_text = decode_vocabulary(model, vocabulary)
    print(repr(id_to_text[vocabulary["Ġthe"]]))   # expect ' the'
    print(repr(id_to_text[vocabulary["Ċ"]]))      # expect '\n'
    print(repr(id_to_text[vocabulary["Ġ,"]]))   # expect ' ,'
    print(repr(id_to_text[vocabulary["Ġ."]]))   # expect ' .'
    print(len(id_to_text), "of", len(vocabulary), "tokens kept")

    dropped = [t for t, i in vocabulary.items() if i not in id_to_text]
    print(dropped[:10])



    

    