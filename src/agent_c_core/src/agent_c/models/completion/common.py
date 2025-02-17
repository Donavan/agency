from pydantic import Field
from typing import Optional

from agent_c.models.base import BaseModel

class CommonCompletionParams(BaseModel):
    model_name: str = Field(..., description="The name of the model to use for the interaction")
    temperature: Optional[float] = Field(None, description="The temperature to use for the interaction, do not combine with top_p")
    max_tokens: Optional[int] = Field(None, description="The maximum number of tokens to generate, defaults to backend defaults")
    user_name: Optional[str] = Field(None, description="The name of the user interacting with the agent")
