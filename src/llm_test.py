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


def get_text_to_ids(id_to_text):
    text_to_ids: dict[str, list[int]] = {}
    for token_id, text in id_to_text.items():
        if text not in text_to_ids:
            text_to_ids[text] = []
        text_to_ids[text].append(token_id)
    return text_to_ids


def build_function_context(prompt: str,
                           functions: list[FunctionDefinition]) -> str:
    catalog = "\n".join(
        json.dumps({"name": f.name, "description": f.description})
        for f in functions
    )
    return (
        "Choose exactly one function from the list.\n\n"
        f"Available functions:\n{catalog}\n\n"
        f"User request: {prompt}\n\n"
        "Name of the correct function:\n"
    )


def allowed_choice_tokens(prefix, choices, text_to_ids):
    allowed: set[int] = set()
    for choice in choices:
        if choice.startswith(prefix):
            rest = choice[len(prefix):]
            for end in range(1, len(rest) + 1):
                allowed.update(text_to_ids.get(rest[:end], []))
    return allowed


def choose_token(logits: list[float], allowed: set[int]) -> int:
    ids = np.array(list(allowed), dtype=np.int64)
    ids = ids[(ids >= 0) & (ids < len(logits))]
    if ids.size == 0:
        raise ValueError("No valid token is available.")
    scores = np.full(len(logits), -np.inf)
    scores[ids] = np.asarray(logits)[ids]
    return int(np.argmax(scores))


def get_function_name(model: Small_LLM_Model, 
                      input_ids: list[int],
                      functions: list[FunctionDefinition],
                      id_to_text: dict[int, str],
                      text_to_ids) -> FunctionDefinition:
    """LLM chooses the function"""
    names = [f.name for f in functions]
    max_name_len = max(len(name) for name in names)
    prefix = ""
    for _ in range(max_name_len + 1):
        # if 
        longer = any(name != prefix and name.startswith(prefix) for name in names)
        if prefix in names and not longer:
            break
        allowed = allowed_choice_tokens(prefix, names, text_to_ids)
        logits = model.get_logits_from_input_ids(input_ids)
        token_id = choose_token(logits, allowed)
        input_ids.append(token_id)
        prefix += id_to_text[token_id]
        print(prefix)
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
    text_to_ids = get_text_to_ids(id_to_text)

    print(len(model.encode("<|im_start|>")[0]))
    prompt = prompts[0]
    context_to_get_function = build_function_context(prompt, functions)
    print(context_to_get_function)
    input_ids = model.encode(context_to_get_function)[0].tolist()
    function = get_function_name(model, input_ids, functions, id_to_text, text_to_ids)
    print(input_ids)
    print(function)
