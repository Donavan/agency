from pydantic import Field
from typing import Union, List, Optional

from agent_c.agents.gpt import GPTCompletionParams
from agent_c.models.interaction.input import BaseInput
from agent_c.models.completion.common import CommonCompletionParams
from agent_c.agents.claude import ClaudeCompletionParams
from agent_c.agents.factory.agent_factory import AgentFactoryBackend
from agent_c.models.interaction.requests.base import BaseInteractRequest

class InteractionStartRequest(BaseInteractRequest):
    interaction_id: Optional[str] = Field(None, description="The ID of the interaction to start")
    backend: AgentFactoryBackend.Name = Field("gpt",
                                                     description="The backend to use for the agent, "
                                                                  "Default is 'gpt'")
    completion_params: Union[CommonCompletionParams, GPTCompletionParams, ClaudeCompletionParams] = Field(..., description="The completion parameters for your backend")
    user_input: List[BaseInput] = Field(..., description="The input from the user to the agent")
    property_bag: dict = Field({}, description="A dictionary of properties to pass to the agent runtime")

class InteractionCancelRequest(BaseInteractRequest):
    interaction_id: str = Field(..., description="The ID of the interaction to cancel")
