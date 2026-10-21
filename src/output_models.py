from pydantic import BaseModel
from typing import Any


class FunctionCall(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, Any]
