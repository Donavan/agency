import os
import json
import openai
import asyncio
import logging

from pydantic import Field
from collections import defaultdict
from tiktoken import Encoding, encoding_for_model
from openai.types.chat import  ChatCompletionChunk
from typing import Any, Dict, List, Union, Optional
from openai import AsyncOpenAI, AsyncStream, AsyncAzureOpenAI

from agent_c.agents.base import BaseAgent
from agent_c.models.events import ToolCallEvent
from agent_c.util.token_counter import TokenCounter
from agent_c.models.interaction.input import AudioInput
from agent_c.models.interaction.input import ImageInput
from agent_c.chat.session_manager import ChatSessionManager
from agent_c.models.events.chat import ReceivedAudioDeltaEvent, InteractionEvent, CompletionEvent
from agent_c.models.completion.common import CommonCompletionParams


class GPTCompletionParams(CommonCompletionParams):
    """
    Vendor specific parameters for interacting with the GPT agent.
    """
    tool_choice: Optional[Union[str, dict]] = Field(None, description="The tool choice to use for the interaction, See OpenAI API docs for details")
    voice:  Optional[str] = None
    presence_penalty: Optional[float] = Field(0, description="Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.")
    seed: Optional[int] = Field(None, description="This feature is in Beta. If specified, Open AI will make a best effort to sample deterministically, such that repeated requests with the same seed and parameters should return the same result. Determinism is not guaranteed")
    service_tier: Optional[str] = Field(None, description="See Open AI Docs for details")
    stop: Optional[Union[str, List[str]]] = Field(None, description="Up to 4 sequences where the API will stop generating further tokens.")

    def __init__(self, **data: Any) -> None:
        if 'model_name' not in data:
            data['model_name'] = os.environ.get("GPT_INTERACTION_MODEL", "gpt-4o")

        super().__init__(**data)

class TikTokenTokenCounter(TokenCounter):
    """
    This is a token counter that uses the TikToken Encoding model to count tokens.
    """

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.encoder: Encoding = encoding_for_model(model_name)

    def count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))

class GPTChatAgent(BaseAgent):
    REASONING_MODELS: List[str] = ["o1", 'o1-mini', 'o3', 'o3-mini']

    def __init__(self, **kwargs) -> None:
        """
        Initialize ChatAgent object.
        """
        if 'max_delay' not in kwargs:
            kwargs['max_delay'] = int(os.environ.get("GPT_AGENT_MAX_DELAY", '10'))

        if "concurrency_limit" not in kwargs:
            kwargs['concurrency_limit'] = int(os.environ.get("GPT_AGENT_CONCURRENCY_LIMIT", '3'))

        super().__init__(**kwargs)
        # Temporary until all the models support this
        if self.model_name in self.__class__.REASONING_MODELS:
            self.root_message_role = "developer"

    @classmethod
    def default_completion_params(cls) -> GPTCompletionParams:
        return GPTCompletionParams()

    @property
    def token_counter(self) -> TokenCounter:
        if self._token_counter is None:
            self._token_counter = TikTokenTokenCounter()

        return self._token_counter

    @classmethod
    def default_client(cls):
        return AsyncOpenAI()

    def _generate_multi_modal_user_message(self, user_input: str, images: List[ImageInput], audio_clips: List[AudioInput]) -> Union[List[dict[str, Any]], None]:
        contents = []
        if user_input is not None and len(user_input) > 0:
            contents.append({"type": "text", "text": user_input})

        for image in images:
            url: Union[str, None] = image.url

            if url is None and image.content is not None:
                url = f"data:{image.content_type};base64,{image.content}"

            if url is not None:
                contents.append({"type": "image_url", "image_url": {"url": url}})

        for clip in audio_clips:
            contents.append({"type": "input_audio", "input_audio": {"data": clip.content, 'format': clip.format}})

        return [{"role": "user", "content": contents}]



    async def __chat_interaction_setup(self, **kwargs) -> dict[str, Any]:
        json_mode: bool = kwargs.get("json_mode", False)
        model_name: str = kwargs.get("model_name", self.model_name)
        kwargs['prompt'] = kwargs.get('prompt', self.prompt)
        sys_prompt: str = await self._render_system_prompt(**kwargs)
        temperature: float = kwargs.get("temperature", self.temperature)
        max_tokens: Optional[int] = kwargs.get("max_tokens", None)
        tool_choice: str = kwargs.get("tool_choice", "auto")
        voice: Optional[str] = kwargs.get("voice", None)
        tool_chest = kwargs.get("tool_chest", self.tool_chest)

        messages = await self._construct_message_array(system_prompt=sys_prompt, **kwargs)

        functions: List[Dict[str, Any]] = tool_chest.active_open_ai_schemas

        if model_name in self.__class__.REASONING_MODELS:
            reasoning_effort = kwargs.get("reasoning_effort", "medium")
            completion_opts = {"model": model_name, "messages": messages, "stream": True, "reasoning_effort": reasoning_effort}
        else:
            completion_opts = {"model": model_name, "temperature": temperature, "messages": messages, "stream": True}
            if voice is not None:
                completion_opts['modalities'] = ["text", "audio"]
                completion_opts['audio'] = {"voice": voice, "format": "pcm16"}

        completion_opts['stream_options'] = {"include_usage": True}

        if len(functions):
            completion_opts['tools'] = functions
            completion_opts['tool_choice'] = tool_choice


        if max_tokens is not None:
            completion_opts['max_completion_tokens'] = max_tokens

        if json_mode:
            completion_opts['response_format'] = {"type": "json_object"}

        user = kwargs.get("user_name", None)
        if user is not None:
            completion_opts["user"] = user

        return {'completion_opts':completion_opts, 'callback_opts': self._callback_opts(**kwargs),
                'tool_chest': tool_chest}

    async def one_shot(self, **kwargs) -> str:
        """
               Perform the chat operation as a one shot.

               Parameters:
               temperature: float, default is self. temperature
                   Controls randomness. Low temperature results in more focused and deterministic output.
               model_name: str, default is self.model_name
                   The language model to be used.
               prompt_metadata: dict, default is empty dict
                   Metadata to be used for rendering the prompt via the PromptBuilder
               session_manager: ChatSessionManager, default is None
                   A session manager to use for the message memory / session info
               session_id: str, default is "none"
                   The session id to use for the chat, if you haven't provided a session manager.
               agent_role: str, default is "assistant"
                   The role of the agent in the chat event stream
               messages: List[dict[str, Any]], default is None
                   A list of messages to use for the chat.
                   If this is not provided, but a session manager is, the messages will be constructed from the session.
               output_format: str, default is "raw"
                   The format to signal the client to expect

               Returns: The text of the model response.
               """
        kwargs['output_format'] = kwargs.get('output_format', 'raw')
        messages = await self.chat(**kwargs)
        return messages[-1]['content']

    async def _save_audio_interaction_to_session(self, mgr: ChatSessionManager, audio_id, transcript: str):
        await self._save_message_to_session(mgr, transcript, "assistant")
        return {"role": "assistant", "audio": {"id": audio_id}}

    async def chat(self, **kwargs) -> List[dict[str, Any]]:
        """
        Perform the chat operation.

        Parameters:
        temperature: float, default is self. temperature
            Controls randomness. Low temperature results in more focused and deterministic output.
        model_name: str, default is self.model_name
            The language model to be used.
        prompt_metadata: dict, default is empty dict
            Metadata to be used for rendering the prompt via the PromptBuilder
        session_manager: ChatSessionManager, default is None
                   A session manager to use for the message memory / session info
       session_id: str, default is "none"
           The session id to use for the chat, if you haven't provided a session manager.
       agent_role: str, default is "assistant"
           The role of the agent in the chat event stream
       messages: List[dict[str, Any]], default is None
           A list of messages to use for the chat.
           If this is not provided, but a session manager is, the messages will be constructed from the session.
        agent_role: str, default is "assistant"
            The role of the agent in the chat event stream
        output_format: str, default is "markdown"
            The format to signal the client to expect

        Returns: A list of messages from the chat.
        """
        opts = await self.__chat_interaction_setup(**kwargs)
        messages: List[dict[str, str]] = opts['completion_opts']['messages']
        session_manager: Union[ChatSessionManager, None] = kwargs.get("session_manager", None)
        tool_chest = opts['tool_chest']
        interacting: bool = True

        delay = 1  # Initial delay between retries

        async with self.semaphore:
            interaction_id = await self._raise_interaction_start(**opts['callback_opts'])
            while interacting and delay < self.max_delay:
                try:
                    tool_calls = []
                    stop_reason: Optional[str] = None
                    audio_id: Optional[str] = None
                    await self._raise_completion_start(opts["completion_opts"], **opts['callback_opts'])
                    response: AsyncStream[ChatCompletionChunk] = await self.client.chat.completions.create(**opts['completion_opts'])

                    collected_messages: List[str] = []
                    try:
                        async for chunk in response:

                            if chunk.choices is None:
                                continue

                            if len(chunk.choices) == 0:
                                # If there are no choices, we're done receiving chunks
                                input_tokens = chunk.usage.prompt_tokens
                                output_tokens = chunk.usage.completion_tokens
                                await self._raise_completion_end(opts["completion_opts"], stop_reason=stop_reason,
                                                                 input_tokens=input_tokens, output_tokens=output_tokens,
                                                                 **opts['callback_opts'])

                                if stop_reason == 'tool_calls' or ((stop_reason=='stop' or stop_reason is None) and len(tool_calls)):
                                    # We have tool calls to make
                                    await self._raise_tool_call_start(tool_calls, vendor="open_ai", **opts['callback_opts'])

                                    try:
                                        # Execute the tool calls
                                        result_messages = await self.__tool_calls_to_messages(tool_calls, tool_chest)
                                    except Exception as e:
                                        logging.exception(f"Failed calling toolsets {e}")
                                        result_messages = []
                                        await self._raise_tool_call_end(tool_calls, result_messages,
                                                                        vendor="open_ai", **opts['callback_opts'])

                                        await self._raise_system_event(f"An error occurred while processing tool calls. {e}", )

                                    if len(result_messages):
                                        await self._raise_tool_call_end(tool_calls, result_messages[1:],
                                                                        vendor="open_ai", **opts['callback_opts'])
                                        messages.extend(result_messages)
                                        await self._raise_history_event(messages, **opts['callback_opts'])
                                else:
                                    # The model has finished, it's side of the interaction so it's time to exit
                                    output_text = "".join(collected_messages)
                                    if audio_id is not None:
                                        messages.append(await self._save_audio_interaction_to_session(session_manager, audio_id, output_text))
                                    else:
                                        messages.append(await self._save_interaction_to_session(session_manager, output_text))
                                    await self._raise_history_event(messages, **opts['callback_opts'])
                                    await self._raise_interaction_end(id=interaction_id, **opts['callback_opts'])
                                    interacting = False
                            else:
                                first_choice = chunk.choices[0]
                                # If we have a finish reason record it and break for the usage payload to be sent
                                # before taking any real action
                                if first_choice.finish_reason is not None:
                                    stop_reason = first_choice.finish_reason
                                    continue

                                # Anything else is either a delta for a tool call or a message
                                if first_choice.delta.tool_calls is not None:
                                    self.__handle_tool_use_fragment(first_choice.delta.tool_calls[0], tool_calls)
                                elif first_choice.delta.content is not None:
                                    collected_messages.append(first_choice.delta.content)
                                    await self._raise_text_delta(first_choice.delta.content, **opts['callback_opts'])
                                elif first_choice.delta.model_extra.get('audio', None) is not None:
                                    audio_delta = first_choice.delta.model_extra['audio']
                                    if audio_id is None:
                                        audio_id = audio_delta.get('id', None)

                                    transcript = audio_delta.get('transcript', None)
                                    if transcript is not None:
                                        collected_messages.append(transcript)
                                        await self._raise_text_delta(transcript, **opts['callback_opts'])

                                    b64_audio = audio_delta.get('data', None)
                                    if b64_audio is not None:
                                        await self._raise_event(ReceivedAudioDeltaEvent(content_type="audio/L16", id=audio_id,
                                                                                        content=b64_audio, **opts['callback_opts']))
                                    elif transcript is None:
                                        #logging.error("No audio data found in response")
                                        continue

                    except openai.APIError as e:
                        logging.exception("OpenAIError occurred while streaming responses: %s", e)
                        await self._raise_system_event(f"An error occurred while streaming responses. {e}\n. Backing off delay at {delay} seconds.",
                                              **opts['callback_opts'])
                        if delay >= self.max_delay:
                            raise
                        await self._exponential_backoff(delay)
                        delay *= 2

                except openai.BadRequestError as e:
                    logging.exception("Invalid request occurred: %s", e)
                    await self._raise_system_event(f"Invalid request error, see logs for details.", **opts['callback_opts'])
                    for message in messages:
                        print(json.dumps(message))
                    await self._raise_completion_end(opts["completion_opts"], stop_reason="exception", **opts['callback_opts'])
                    raise
                except openai.APITimeoutError as e:
                    await self._raise_system_event(f"Open AI Timeout error.  Will retry in {delay} seconds", **opts['callback_opts'])
                    logging.exception("OpenAIError occurred: %s", e)
                    if delay >= self.max_delay:
                        raise
                    await self._exponential_backoff(delay)
                    delay *= 2
                except openai.InternalServerError as e:
                    logging.exception("Open AI InternalServerError occurred: %s", e)
                    await self._raise_system_event(f"Open AI internal server error.  Will retry in {delay} seconds", **opts['callback_opts'])
                    if delay >= self.max_delay:
                        raise
                    await self._exponential_backoff(delay)
                    delay *= 2
                except Exception as e:
                    s = e.__str__()
                    logging.exception("Error occurred during chat completion: %s", e)
                    await self._raise_system_event(f"Exception in chat completion {s}", **opts['callback_opts'])
                    await self._raise_completion_end(opts["completion_opts"], stop_reason="exception", **opts['callback_opts'])
                    raise

        return messages

    @staticmethod
    def _translate_completion_params(completion_params:Union[GPTCompletionParams, CommonCompletionParams]) -> dict[str, Any]:
        """Convert a completion params object to a dictionary of options for the OpenAI API"""
        opts = completion_params.model_dump(exclude_none=True, exclude={"type"})
        opts['stream_options'] = {"include_usage": True}
        opts['stream']: True

        if 'user_name' in opts:
            opts['user'] = opts.pop('user_name')

        if 'voice' in opts:
            opts['modalities'] = ["text", "audio"]
            opts['audio'] = {"voice": opts.pop('voice'), "format": "pcm16"}

        if 'max_tokens' in opts:
            opts['max_completion_tokens'] = opts.pop('max_tokens')

        return opts

    async def interact(self, session_id: str,  interaction_id: str, input_queue: asyncio.Queue, output_queue: asyncio.Queue,
                       completion_params:Union[GPTCompletionParams, CommonCompletionParams], messages: list):

        opts = self._translate_completion_params(completion_params)
        interacting: bool = True
        pending_tool_results: Optional[List] = None
        client_wants_cancel: bool = False

        async def process_queue():
            nonlocal interacting, pending_tool_results, client_wants_cancel, self
            while interacting:
                try:
                    event = await input_queue.get()
                    if event.type == "interaction_cancel":
                        client_wants_cancel = True
                        interacting = False

                    input_queue.task_done()
                except Exception as e:
                    logging.error(f"Error processing input queue: {e}")

        delay = 1  # Initial delay between retries

        async with self.semaphore:
            await output_queue.put(InteractionEvent(started=True, id=interaction_id, session_id=session_id))
            while interacting and delay < self.max_delay and not client_wants_cancel:
                try:
                    tool_calls = []
                    stop_reason: Optional[str] = None
                    audio_id: Optional[str] = None

                    await output_queue.put(CompletionEvent(completion_options=opts, running=True, session_id=session_id))
                    opts['messages'] = messages

                    response: AsyncStream[ChatCompletionChunk] = await self.client.chat.completions.create(**opts)

                    collected_messages: List[str] = []
                    try:
                        async for chunk in response:

                            if chunk.choices is None:
                                continue

                            if len(chunk.choices) == 0:
                                # If there are no choices, we're done receiving chunks
                                input_tokens = chunk.usage.prompt_tokens
                                output_tokens = chunk.usage.completion_tokens
                                await output_queue.put(CompletionEvent(completion_options=opts, running=False, session_id=session_id,
                                                                       stop_reason=stop_reason, input_tokens=input_tokens, output_tokens=output_tokens))

                                if stop_reason == 'tool_calls' or ((stop_reason=='stop' or stop_reason is None) and len(tool_calls)):
                                    # We have tool calls to make
                                    await output_queue.put(ToolCallEvent(tool_calls=tool_calls, vendor="open_ai", session_id=session_id,
                                                                         active=True, interaction_id=interaction_id))
                                    try:
                                        # Execute the tool calls
                                        result_messages = await self.__tool_calls_to_messages(tool_calls, tool_chest)
                                    except Exception as e:
                                        logging.exception(f"Failed calling toolsets {e}")
                                        result_messages = []
                                        await self._raise_tool_call_end(tool_calls, result_messages,
                                                                        vendor="open_ai", **opts['callback_opts'])

                                        await self._raise_system_event(f"An error occurred while processing tool calls. {e}", )

                                    if len(result_messages):
                                        await self._raise_tool_call_end(tool_calls, result_messages[1:],
                                                                        vendor="open_ai", **opts['callback_opts'])
                                        messages.extend(result_messages)
                                        await self._raise_history_event(messages, **opts['callback_opts'])
                                else:
                                    # The model has finished, it's side of the interaction so it's time to exit
                                    output_text = "".join(collected_messages)
                                    if audio_id is not None:
                                        messages.append(await self._save_audio_interaction_to_session(session_manager, audio_id, output_text))
                                    else:
                                        messages.append(await self._save_interaction_to_session(session_manager, output_text))
                                    await self._raise_history_event(messages, **opts['callback_opts'])
                                    await self._raise_interaction_end(id=interaction_id, **opts['callback_opts'])
                                    interacting = False
                            else:
                                first_choice = chunk.choices[0]
                                # If we have a finish reason record it and break for the usage payload to be sent
                                # before taking any real action
                                if first_choice.finish_reason is not None:
                                    stop_reason = first_choice.finish_reason
                                    continue

                                # Anything else is either a delta for a tool call or a message
                                if first_choice.delta.tool_calls is not None:
                                    self.__handle_tool_use_fragment(first_choice.delta.tool_calls[0], tool_calls)
                                elif first_choice.delta.content is not None:
                                    collected_messages.append(first_choice.delta.content)
                                    await self._raise_text_delta(first_choice.delta.content, **opts['callback_opts'])
                                elif first_choice.delta.model_extra.get('audio', None) is not None:
                                    audio_delta = first_choice.delta.model_extra['audio']
                                    if audio_id is None:
                                        audio_id = audio_delta.get('id', None)

                                    transcript = audio_delta.get('transcript', None)
                                    if transcript is not None:
                                        collected_messages.append(transcript)
                                        await self._raise_text_delta(transcript, **opts['callback_opts'])

                                    b64_audio = audio_delta.get('data', None)
                                    if b64_audio is not None:
                                        await self._raise_event(ReceivedAudioDeltaEvent(content_type="audio/L16", id=audio_id,
                                                                                        content=b64_audio, **opts['callback_opts']))
                                    elif transcript is None:
                                        #logging.error("No audio data found in response")
                                        continue

                    except openai.APIError as e:
                        logging.exception("OpenAIError occurred while streaming responses: %s", e)
                        await self._raise_system_event(f"An error occurred while streaming responses. {e}\n. Backing off delay at {delay} seconds.",
                                              **opts['callback_opts'])
                        if delay >= self.max_delay:
                            raise
                        await self._exponential_backoff(delay)
                        delay *= 2

                except openai.BadRequestError as e:
                    logging.exception("Invalid request occurred: %s", e)
                    await self._raise_system_event(f"Invalid request error, see logs for details.", **opts['callback_opts'])
                    for message in messages:
                        print(json.dumps(message))
                    await self._raise_completion_end(opts["completion_opts"], stop_reason="exception", **opts['callback_opts'])
                    raise
                except openai.APITimeoutError as e:
                    await self._raise_system_event(f"Open AI Timeout error.  Will retry in {delay} seconds", **opts['callback_opts'])
                    logging.exception("OpenAIError occurred: %s", e)
                    if delay >= self.max_delay:
                        raise
                    await self._exponential_backoff(delay)
                    delay *= 2
                except openai.InternalServerError as e:
                    logging.exception("Open AI InternalServerError occurred: %s", e)
                    await self._raise_system_event(f"Open AI internal server error.  Will retry in {delay} seconds", **opts['callback_opts'])
                    if delay >= self.max_delay:
                        raise
                    await self._exponential_backoff(delay)
                    delay *= 2
                except Exception as e:
                    s = e.__str__()
                    logging.exception("Error occurred during chat completion: %s", e)
                    await self._raise_system_event(f"Exception in chat completion {s}", **opts['callback_opts'])
                    await self._raise_completion_end(opts["completion_opts"], stop_reason="exception", **opts['callback_opts'])
                    raise

        return messages


    @staticmethod
    def __handle_tool_use_fragment(tool_call, tool_calls):
        index = tool_call.index
        if index == len(tool_calls):
            tool_calls.append(defaultdict(str))
        if tool_call.id:
            tool_calls[index]['id'] = tool_call.id
        if tool_call.function:
            if tool_call.function.name:
                tool_calls[index]['name'] = tool_call.function.name
            if tool_call.function.arguments:
                tool_calls[index]['arguments'] += (tool_call.function.arguments)

    async def __tool_calls_to_messages(self, tool_calls, tool_chest):
        async def make_call(tool_call):
            fn = tool_call['name'] #.replace(".", Toolset.tool_sep)  # The agent sometimes messes this up
            args = json.loads(tool_call['arguments'])
            ai_call = {"id": tool_call['id'],
                       "function": {"name": fn, "arguments": tool_call['arguments']},
                       'type': 'function'
                       }
            try:
                function_response = await self._call_function(tool_chest, fn, args)

                call_resp = {"role": "tool", "tool_call_id": tool_call['id'], "name": fn,
                             "content": function_response}

            except Exception as e:
                call_resp = {"role": "tool", "tool_call_id": tool_call['id'], "name": fn,
                             "content": f"Exception: {e}"}

            return ai_call, call_resp

        # Schedule all the calls concurrently
        tasks = [make_call(tool_call) for tool_call in tool_calls]
        completed_calls = await asyncio.gather(*tasks)

        # Unpack the resulting ai_calls and resp_calls
        ai_calls, results = zip(*completed_calls)

        return [{'role': 'assistant', 'tool_calls': list(ai_calls), 'content': ''}] + list(results)

class AzureOpenAIChatAgent(GPTChatAgent):
    @classmethod
    def default_client(cls):
        if os.environ.get("AZURE_OPENAI_ENDPOINT", None) is not None and os.environ.get("AZURE_OPENAI_API_KEY", None) is not None:
            return AsyncAzureOpenAI(azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                                    api_key=os.environ["AZURE_OPENAI_API_KEY"],
                                    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"))
        else:
            raise ValueError("AZURE_OPENAI_ENDPOINT and/or AZURE_OPENAI_API_KEY not found in environment variables. (might wanna check AZURE_OPENAI_API_VERSION as well)")