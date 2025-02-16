from agent_c.models.interaction.input import BaseInput

class TextInput(BaseInput):
    """Model representing text input with content.

    Attributes:
        content (str): The actual content of the text input
    """
    content: str
