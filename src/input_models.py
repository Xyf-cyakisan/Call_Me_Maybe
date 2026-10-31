from pydantic import BaseModel, Field, ConfigDict
from typing import Annotated, Literal

NonEmptyString = Annotated[str, Field(min_length=1)]


class Type(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["boolean", "integer", "string", "number"]


class Function(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[NonEmptyString, Type]
    returns: Type


class Prompt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)
