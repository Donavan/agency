from typing import Optional

from pydantic import Field

from agent_c.util import to_snake_case
from agent_c.models.events.base import BaseEvent

class BaseInteractRequest(BaseEvent):
    """
    Base class for all interaction requests.
    """
    session_id: str = Field(..., description="The session ID for the interaction")
    role: Optional[str] = Field(None, description="The role of that initiated the request")

    def __init__(self, **data) -> None:
        if 'type' not in data:
            data['type'] = to_snake_case(self.__class__.__name__.removesuffix('Request'))

        super().__init__(**data)
