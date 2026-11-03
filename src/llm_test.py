#!/usr/bin/env python3

import json
from pydantic import BaseModel, ValidationError
from llm_sdk.llm_sdk import Small_LLM_Model
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
        raise ValueError("Invalid JSON in function calling tests file:", path)
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


def get_number_ids(id_to_text: dict[int, str]) -> set[int]:
    """Ids of tokens made only of digits, '.' and '-'."""
    return {i for i, t in id_to_text.items()
            if all(c in "0123456789.-" for c in t)}


def get_string_ids(id_to_text: dict[int, str]) -> set[int]:
    """Tokens that can be part of a string value
    (no quote, backslash, newline)."""
    return {i for i, t in id_to_text.items() if not any(c in t for c in '\n')}


def build_function_context(prompt: str,
                           functions: list[FunctionDefinition]) -> str:
    catalog = "\n".join(
        json.dumps({"name": f.name,
                    "description": f.description,
                    "parameters": {n: p.type for n, p in f.parameters.items()}
                    })
        for f in functions
    )
    # return (
    #     "Choose exactly one function from the list.\n\n"
    #     f"Available functions:\n{catalog}\n\n"
    #     f"User request: {prompt}\n\n"
    #     "Function call as JSON:\n"
    #     '{"name": "'
    # )
    return (
        "<|im_start|>system\n"
        "You translate user requests into function calls. "
        "Choose exactly one function from the list and fill its parameters "
        "with the exact values the function needs (e.g. real regex "
        "patterns, real characters).\n\n"
        f"Available functions:\n{catalog}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        '{"name": "'
    )


def allowed_choice_tokens(prefix, choices, text_to_ids):
    allowed: set[int] = set()
    for choice in choices:
        if choice.startswith(prefix):
            rest = choice[len(prefix):]
            for end in range(1, len(rest) + 1):
                allowed.update(text_to_ids.get(rest[:end], []))
    return allowed


def choose_token(logits: list[float], allowed: set[int], id_to_text) -> int:
    ids = np.array(list(allowed), dtype=np.int64)
    ids = ids[(ids >= 0) & (ids < len(logits))]
    if ids.size == 0:
        raise ValueError("No valid token is available.")
    scores = np.full(len(logits), -np.inf)
    scores[ids] = np.asarray(logits)[ids]
    # test______________________________
    # best = ids[np.argsort(scores[ids])[::-1][:5]]
    # for token_id in best:
    # print(repr(id_to_text[int(token_id)]),
    # round(scores[token_id], 2))
    return int(np.argmax(scores))


def get_function_name(model: Small_LLM_Model,
                      input_ids: list[int],
                      functions: list[FunctionDefinition],
                      id_to_text: dict[int, str],
                      text_to_ids) -> FunctionDefinition:
    """LLM chooses the function"""
    if not functions:
        raise ValueError("No functions available.")
    names = [f.name + '"' for f in functions]
    max_name_len = max(len(name) for name in names)
    prefix = ""
    for _ in range(max_name_len):
        if prefix in names:
            break
        allowed = allowed_choice_tokens(prefix, names, text_to_ids)
        logits = model.get_logits_from_input_ids(input_ids)
        token_id = choose_token(logits, allowed, id_to_text)
        input_ids.append(token_id)
        prefix += id_to_text[token_id]
    for function in functions:
        if function.name + '"' == prefix:
            return function
    raise ValueError(f"Model produced an invalid function name: {prefix}")


def extract_numeric_param(model, input_ids,
                          id_to_text, number_ids, number_end_ids):
    """ """
    prefix_re = r"-?(\d+(\.\d*)?)?"
    complete_re = r"-?\d+(\.\d+)?"
    number = ""
    for _ in range(100):
        allowed = {i for i in number_ids
                   if re.fullmatch(prefix_re, number + id_to_text[i])}
        if re.fullmatch(complete_re, number):
            allowed |= number_end_ids
        logits = model.get_logits_from_input_ids(input_ids)
        token_id = choose_token(logits, allowed, id_to_text)
        if token_id in number_end_ids:
            break
        input_ids.append(token_id)
        number += id_to_text[token_id]
    if not re.fullmatch(complete_re, number):
        raise ValueError(f"Invalid number generated: {number!r}")
    return float(number)


def extract_string_param(model, input_ids,
                         id_to_text, string_ids, string_end_ids):
    """ """
    allowed = string_ids | string_end_ids
    text = ""
    for _ in range(100):
        logits = model.get_logits_from_input_ids(input_ids)
        token_id = choose_token(logits, allowed, id_to_text)
        if token_id in string_end_ids:
            break
        input_ids.append(token_id)
        text += id_to_text[token_id]
    return text


def get_parameters(function: FunctionDefinition,
                   model: Small_LLM_Model,
                   input_ids: list[int],
                   id_to_text: dict[int, str],
                   text_to_ids,
                   number_ids,
                   string_ids,
                   prompt,
                   string_end_ids,
                   number_end_ids):
    params = {}
    for i, (name, param_defenition) in enumerate(function.parameters.items()):
        separator = "" if i == 0 else ", "
        key = f"{separator}{json.dumps(name)}: "
        if param_defenition.type == "string":
            input_ids += model.encode(key + '"')[0].tolist()
            params[name] = extract_string_param(model,
                                                input_ids,
                                                id_to_text,
                                                string_ids,
                                                string_end_ids)
            input_ids += model.encode('"')[0].tolist()
        elif param_defenition.type == "number":
            input_ids += model.encode(key)[0].tolist()
            params[name] = extract_numeric_param(model,
                                                 input_ids,
                                                 id_to_text,
                                                 number_ids,
                                                 number_end_ids)
        # print("__________")
    return params


def test_llm():
    functions = get_function_defenitions(
        "data/input/functions_definition.json")
    prompts = get_function_calling_tests(
        "data/input/function_calling_tests.json")

    model = Small_LLM_Model()

    with open(model.get_path_to_vocab_file(), encoding="utf-8") as f:
        vocabulary = json.load(f)

    id_to_text = decode_vocabulary(model, vocabulary)
    text_to_ids = get_text_to_ids(id_to_text)
    number_ids = get_number_ids(id_to_text)
    string_ids = get_string_ids(id_to_text)

    def get_end_ids(texts: list[str], text_to_ids) -> set[int]:
        """Ids of the tokens that close a value."""
        return {i for t in texts for i in text_to_ids.get(t, [])}

    STRING_END = ['"', '",', '"}', '"}}', '",\n', '"\n']
    NUMBER_END = [",", "}", " }", "},", "}}", ",\n", "}\n"]
    string_end_ids = get_end_ids(STRING_END, text_to_ids)
    number_end_ids = get_end_ids(NUMBER_END, text_to_ids)

    for prompt in prompts:
        # print(f"{prompt.prompt}")
        # print("_______________")
        context = build_function_context(prompt.prompt, functions)
        input_ids = model.encode(context)[0].tolist()
        function = get_function_name(model,
                                     input_ids,
                                     functions,
                                     id_to_text,
                                     text_to_ids)
        print(f"{prompt.prompt} -> {function.name}")
        input_ids += model.encode(', "parameters": {')[0].tolist()
        params = get_parameters(function,
                                model,
                                input_ids,
                                id_to_text,
                                text_to_ids,
                                number_ids,
                                string_ids,
                                prompt.prompt,
                                string_end_ids,
                                number_end_ids)
        print(params)
        print("________________________")
