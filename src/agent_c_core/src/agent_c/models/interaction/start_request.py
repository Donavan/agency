from pydantic import Field
from typing import Union, List

from agent_c.models.base import BaseModel
from agent_c.agents.gpt import GPTCompletionParams
from agent_c.models.interaction.input import BaseInput
from agent_c.agents.claude import ClaudeCompletionParams

class InteractionStartRequest(BaseModel):
    completion_prompt: Union[GPTCompletionParams, ClaudeCompletionParams]
    user_input: List[BaseInput] = Field(..., description="The input from the user to the agent")
