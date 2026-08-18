from pydantic import BaseModel
from llm_sdk.llm_sdk import encode, decode, get_logits_from_input_ids, get_path_to_vocabulary_json


class FunctionCalling(BaseModel):
    promt: str


class ParameterDefenition(BaseModel):
    data_type: str # string, number, 

class FunctionDefenition(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParameterDefenition]
    returns: str

