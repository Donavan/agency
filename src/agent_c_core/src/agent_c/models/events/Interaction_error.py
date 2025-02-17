from agent_c.models.events.session_event import SemiSessionEvent
from pydantic import Field

class InteractionErrorEvent(SemiSessionEvent):
    """
    An error that occurred during an interaction.
    """
    error_message: str = Field(..., description="The error message that occurred during the interaction")

    def __init__(self, **data) -> None:
        super().__init__(**data)
