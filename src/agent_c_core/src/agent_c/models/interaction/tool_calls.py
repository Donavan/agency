from typing import Any, Self

from pydantic import Field

from agent_c import BaseModel
from agent_c.models.events.interaction import BaseInteractionEvent

class CommonToolCall(BaseModel):
    id: str = Field(..., description="The ID of the tool call")
    name: str = Field(..., description="The name of the tool.")
    arguments: dict = Field({}, description="The arguments to pass to the tool.")


class CommonAgentToolCallsEvent(BaseInteractionEvent):
    pass