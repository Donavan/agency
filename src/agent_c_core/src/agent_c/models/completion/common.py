from pydantic import Field
from typing import Optional, Any

from agent_c.util import to_snake_case
from agent_c.models.base import BaseModel


class CommonCompletionParams(BaseModel):
    type: str = Field(..., description="The type of the completion params.")
    model: str = Field(..., description="The name of the model to use for the interaction")
    temperature: Optional[float] = Field(None, description="The temperature to use for the interaction, do not combine with top_p")
    max_tokens: Optional[int] = Field(None, description="The maximum number of tokens to generate, defaults to backend defaults")
    user_name: Optional[str] = Field(None, description="The name of the user interacting with the agent")

    def __init__(self, **data: Any) -> None:
        if 'type' not in data:
            data['type'] = to_snake_case(self.__class__.__name__.removesuffix('CompletionParams'))

        super().__init__(**data)