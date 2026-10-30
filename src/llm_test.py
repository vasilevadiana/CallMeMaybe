#!/usr/bin/env python3

import json
from pydantic import BaseModel, Field, ValidationError
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
    except FileNotFoundError:
        raise ValueError(f"Function definition file not found: {path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in function definition file: {path}")
    except OSError as exc:
        raise ValueError(f"Cannot read {path}: {exc}")
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array.")
    try:
        return [FunctionDefinition.model_validate(item) for item in data]
    except ValidationError as exc:
        raise ValueError(f"Invalid function definition in {path}: {exc}")


def get_function_calling_tests(path: str) -> list[FunctionCalling]:
    """Load function calling tests from JSON file and validate"""
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError:
        raise ValueError(f"Function calling tests file not found: {path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in function calling tests file: {path}")
    except OSError as exc:
        raise ValueError(f"Cannot read {path}: {exc}")
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array.")
    try:
        return [FunctionCalling.model_validate(item) for item in data]
    except ValidationError as exc:
        raise ValueError(f"Invalid prompt entry in {path}: {exc}")


def decode_vocabulary(model: Small_LLM_Model,
                      vocabulary: dict[str, int]) -> dict[int, str]:
    """Map each token id to its real text using the SDK's decode."""
    id_to_text: dict[int, str] = {}
    for token_id in vocabulary.values():
        text = model.decode([token_id])
        if text and "\ufffd" not in text:
            id_to_text[token_id] = text
    return id_to_text


def build_function_prompt(initial_prompt: str,
                          functions: list[FunctionDefinition]) -> str:
    """Qwen chat prompt that ends where the function name begins."""
    
    NAME_PREFIX = '{"name": "'

    catalog = "\n".join(
        json.dumps({
            "name": f.name,
            "description": f.description,
            "parameters": {n: p.type for n, p in f.parameters.items()},
        })
        for f in functions
    )
    system = (
        "You are a function calling assistant. Choose the single function "
        "that best fulfils the user request, then fill its parameters. "
        "Copy string values exactly as written in the request, without "
        "surrounding quotes. Write numbers as digits. "
        'Answer only with JSON: {"name": ..., "parameters": {...}}.'
        "\n\nAvailable functions:\n" + catalog
    )
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{initial_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n" + NAME_PREFIX
    )


def get_function_name(model: Small_LLM_Model, 
                      input_ids: list[int],
                      functions: list[FunctionDefinition],
                      id_to_text: dict[int, str]) -> FunctionDefinition:
    """The LLM chooses the function; output is forced to a real name."""
    names = [f.name for f in functions]
    prefix = ""
    for _ in range(max(len(n) for n in names) + 1):
        longer = any(n != prefix and n.startswith(prefix) for n in names)
        if prefix in names and not longer:
            break
        logits = model.get_logits_from_input_ids(input_ids)
        token_id = int(np.argmax(logits))
        input_ids.append(token_id)
        prefix += id_to_text[token_id]
    for function in functions:
        if function.name == prefix:
            return function


def test_llm():
    print("############################################## FUNC_NAME TEST #################################################")
    functions = get_function_defenitions("data/input/functions_definition.json")
    prompts = get_function_calling_tests("data/input/function_calling_tests.json")

    model = Small_LLM_Model()

    with open(model.get_path_to_vocab_file(), encoding="utf-8") as f:
        vocabulary = json.load(f)

    id_to_text = decode_vocabulary(model, vocabulary)

    print(len(model.encode("<|im_start|>")[0]))
    prompt = prompts[0]
    function_prompt = build_function_prompt(prompt, functions)
    print(function_prompt)
    input_ids = model.encode(function_prompt)[0].tolist()
    function = get_function_name(model, input_ids, functions, id_to_text)
    print(input_ids)
    print(function)
