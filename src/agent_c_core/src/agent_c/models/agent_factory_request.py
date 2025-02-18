from pydantic import Field
from typing import Literal, Optional, Any

from agent_c.models.base import BaseModel

class AgentRuntimeParams(BaseModel):
    # Immutable BaseAgent parameters
    backend: Literal["azure_openai", "openai",
                     "claude", "claude_aws"] = Field("gpt",
                                                     description="The backend to use for the agent, "
                                                                 "valid options are 'azure_oai', 'gpt', 'claude', 'claude_aws'. "
                                                                 "Default is 'gpt'")
    concurrency_limit: int = Field(3, description="The maximum number of concurrent "
                                                         "requests to the completion API ths agent can make")

class AgentCreationOptions(BaseModel):
    runtime: AgentRuntimeParams = Field(..., description="The parameters for the backend runtime agents will use")

class ClientAgentCreationOptions(BaseModel):
    backend: Literal["azure_openai", "openai",
    "claude", "claude_aws"] = Field("gpt",
                                           description="The backend to use for the agent, "
                                                       "valid options are 'azure_oai', 'gpt', 'claude', 'claude_aws'. "
                                                       "Default is 'gpt'")



