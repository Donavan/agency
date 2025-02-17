from pydantic import Field
from typing import Union, List

from agent_c.agents.gpt import GPTCompletionParams
from agent_c.models.interaction.input import BaseInput
from agent_c.agents.claude import ClaudeCompletionParams
from agent_c.models.interaction.requests.base import BaseInteractRequest

class InteractionStartRequest(BaseInteractRequest):
    completion_params: Union[GPTCompletionParams, ClaudeCompletionParams] = Field(..., description="The completion parameters for your backend")
    user_input: List[BaseInput] = Field(..., description="The input from the user to the agent")
    property_bag: dict = Field({}, description="A dictionary of properties to pass to the agent runtime")
