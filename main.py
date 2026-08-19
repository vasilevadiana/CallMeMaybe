#!/usr/bin/env python3

from pydantic import BaseModel
from llm_sdk.llm_sdk import encode, decode, get_logits_from_input_ids, get_path_to_vocabulary_json
import json


class FunctionCalling(BaseModel):
    promt: str


class ParameterDefenition(BaseModel):
    data_type: str # string, number, 


class FunctionDefenition(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParameterDefenition]
    returns: str


def get_function_defenitions(path: str)-> list[FunctionDefenition]:
    """
    Load function defenitions from json file
    """
    try:
        with open(path) as f:
            data = json.load(f)
        print(data)
    except FileNotFoundError:
        raise ValueError("Function defenition file not found:", path)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON:", path)