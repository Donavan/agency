from pydantic import Field
from typing import Literal, Optional, Any

from agent_c.models.base import BaseModel

class AgentParams(BaseModel):
    # Immutable BaseAgent parameters
    backend: Literal["gpt", "claude"] = Field("gpt", description="The backend to use for the agent, valid options are 'gpt' and 'claude'.  Default is 'gpt'")
    concurrency_limit: int = Field(3, description="The maximum number of concurrent "
                                                         "requests to the completion API ths agent can make")

    # These establish defaults that can be overridden by during the completion call
    model_name: str = Field(..., description="The name of the model to use")
    temperature: float = Field(0.5, description="The temperature to use for completions, "
                                                       "will be ignored by reasoning models",
                               gt=0, le=1.0)

class AgentFactoryRequest(BaseModel):
    agent_params: AgentParams = Field(..., description="The parameters for the agent to create, beyond the backend selection, mostly to establish default values")



