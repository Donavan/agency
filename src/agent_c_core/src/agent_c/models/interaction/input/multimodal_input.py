from pydantic import Field
from typing import List, Union

from agent_c.models.interaction.input import BaseInput
from agent_c.models.interaction.input import (
    AudioInput,
    FileInput,
    ImageInput,
    TextInput,
    VideoInput,
)



class MultimodalInput(BaseInput):
    """
    A class representing multimodal input that can contain multiple types of content.

    Attributes:
        content (List[Union[TextInput, ImageInput, AudioInput, FileInput, VideoInput]]): 
            A list of different types of input content that can include text, images, 
            audio, files, and videos.
    """

    content: List[Union[TextInput, ImageInput, AudioInput, FileInput, VideoInput]] = Field(
        ...,
        alias="contents"
    )