from pydantic import Field
from typing import Optional

from agent_c.models.events.session_event import SessionEvent

class BaseInteractionEvent(SessionEvent):
    interaction_id: Optional[str] = Field(None, description="The ID of the interaction, MUST be set before being sent to a client")