from typing import List, Union

from pydantic import Field

from agent_c.agents.gpt import GPTCompletionParams
from agent_c.agents.claude import ClaudeCompletionParams
from agent_c.models.completion.common import CommonCompletionParams
from agent_c.models.agent_interface.base import BaseInterfaceRequest, BaseInterfaceResponse



class AvailableBackendsRequest(BaseInterfaceRequest):
    pass

class AvailableBackendsResponse(BaseInterfaceResponse):
    backends: dict[str, Union[CommonCompletionParams, GPTCompletionParams, ClaudeCompletionParams]] = Field(..., description="The list of available backends")
