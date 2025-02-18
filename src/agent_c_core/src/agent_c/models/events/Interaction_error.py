from typing import Optional

from agent_c.models.events.session_event import SemiSessionEvent
from pydantic import Field

class InteractionErrorEvent(SemiSessionEvent):
    """
    An error that occurred during an interaction.
    """
    interation_id: Optional[str] = Field(None, description="The ID of the interaction that the error occurred in, may be None is an id hasn't been established")
    error_message: str = Field(..., description="The error message that occurred during the interaction")

    def __init__(self, **data) -> None:
        super().__init__(**data)
