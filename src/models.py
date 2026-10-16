from pydantic import BaseModel, Field, ConfigDict
from typing import Literal, TypedDict, Annotated

NonEmptyString = Annotated[str, Field(min_length=1)]


class Parameters(TypedDict):
    type: Literal["string", "number", "boolean"]


class Function(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[NonEmptyString, Parameters]
    returns: Parameters


class Prompt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)

    def get_prompt(self) -> str:
        return self.prompt
