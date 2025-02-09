import asyncio
import copy
import logging

from typing import Any, List, Union, Dict
from anthropic import AsyncAnthropic, APITimeoutError, Anthropic

from agent_c.agents.base import BaseAgent
from agent_c.chat.session_manager import ChatSessionManager
from agent_c.models.input.audio_input import AudioInput
from agent_c.models.input.image_input import ImageInput
from agent_c.util.token_counter import TokenCounter

class ClaudeChatAgent(BaseAgent):
    CLAUDE_MAX_TOKENS: int = 8192
    class ClaudeTokenCounter(TokenCounter):

        def __init__(self):
            self.anthropic: Anthropic = Anthropic()

        def count_tokens(self, text: str) -> int:
            response = self.anthropic.messages.count_tokens(
                model="claude-3-5-sonnet-20241022",
                system="",
                messages=[{
                    "role": "user",
                    "content": text
                }],
            )

            return response.input_tokens



    @property
    def token_counter(self) -> TokenCounter:
        if self._token_counter is None:
            self._token_counter = ClaudeChatAgent.ClaudeTokenCounter()

        return self._token_counter

    @classmethod
    def default_client(cls):
        return AsyncAnthropic()

    async def __interaction_setup(self, **kwargs) -> dict[str, Any]:
        model_name: str = kwargs.get("model_name", self.model_name)
        sys_prompt: str = await self._render_system_prompt(**kwargs)
        temperature: float = kwargs.get("temperature", self.temperature)
        max_tokens: int = kwargs.get("max_tokens", self.CLAUDE_MAX_TOKENS)

        messages = await self._construct_message_array(**kwargs)
        callback_opts = self._callback_opts(**kwargs)

        tool_chest = kwargs.get("tool_chest", self.tool_chest)

        functions: List[Dict[str, Any]] = tool_chest.active_claude_schemas

        completion_opts = {"model": model_name, "messages": messages, "system": sys_prompt,  "max_tokens": max_tokens, 'temperature': temperature}

        if len(functions):
            completion_opts['tools'] = functions

        session_manager: Union[ChatSessionManager, None] = kwargs.get("session_manager", None)
        if session_manager is not None:
            completion_opts["metadata"] = {'user_id': session_manager.user.user_id}

        opts = {"callback_opts": callback_opts, "completion_opts": completion_opts, 'tool_chest': tool_chest}
        return opts

    async def chat(self, **kwargs) -> List[dict[str, Any]]:
        """
        Perform the chat operation.

        Parameters:
        session_id: str
            Unique identifier for the session.
        model_name: str, default is self.model_name
            The language model to be used.
        prompt_metadata: dict, default is empty dict
            Metadata to be used for rendering the prompt via the PromptBuilder
        model_name: str, default is self.model_name
            The name of the Claude model to use.
            - Can be one of "opus", "sonnet", or "haiku". In which case ANTHROPIC_MODEL_PATTERN will bs used
              to generate the model name.
            - A full Claude model name can also be used.
        session_manager: ChatSessionManager, default is None
            A session manager to use for the message memory / session info
        session_id: str, default is "none"
            The session id to use for the chat, if you haven't provided a session manager.
        agent_role: str, default is "assistant"
            The role of the agent in the chat event stream
        messages: List[dict[str, Any]], default is None
            A list of messages to use for the chat.
            If this is not provided, but a session manager is, the messages will be constructed from the session.
        output_format: str, default is "markdown"
            The format to signal the client to expect


        Returns: A list of messages.
        """
        opts = await self.__interaction_setup(**kwargs)
        callback_opts = opts["callback_opts"]
        tool_chest = opts['tool_chest']

        session_manager: Union[ChatSessionManager, None] = kwargs.get("session_manager", None)
        messages = opts["completion_opts"]["messages"]


        delay = 1  # Initial delay between retries
        async with self.semaphore:
            interaction_id = await self._raise_interaction_start(**callback_opts)
            while delay <= self.max_delay:
                try:
                    await self._raise_completion_start(opts["completion_opts"], **callback_opts)
                    async with self.client.messages.stream (**opts["completion_opts"]) as stream:
                        collected_messages = []
                        collected_tool_calls = []
                        input_tokens = 0

                        async for event in stream:
                            if event.type == "message_start":
                                input_tokens = event.message.usage.input_tokens

                            elif event.type == 'message_stop':
                                output_tokens = event.message.usage.output_tokens
                                await self._raise_completion_end(opts["completion_opts"], stop_reason=stop_reason, input_tokens=input_tokens, output_tokens=output_tokens, **callback_opts)
                                if stop_reason != 'tool_use':
                                    output_text = "".join(collected_messages)
                                    messages.append(await self._save_interaction_to_session(session_manager, output_text))
                                    await self._raise_history_event(messages, **callback_opts)
                                    await self._raise_interaction_end(id=interaction_id, **callback_opts)
                                    return messages
                                else:
                                    await self._raise_tool_call_start(collected_tool_calls, vendor="anthropic",
                                                                      **callback_opts)
                                    messages.extend(await self.__tool_calls_to_messages(collected_tool_calls, tool_chest))
                                    await self._raise_tool_call_end(collected_tool_calls, messages[-1]['content'],
                                                                    vendor="anthropic", **callback_opts)
                                    await self._raise_history_event(messages, **callback_opts)

                            elif event.type == "content_block_start":
                                content_type = event.content_block.type
                                if content_type == "text":
                                    content = event.content_block.text
                                    if len(content) > 0:
                                        collected_messages.append(content)
                                        await self._raise_text_delta(content, **callback_opts)
                                elif content_type == "tool_use":
                                    tool_call = event.content_block.model_dump()
                                    collected_tool_calls.append(tool_call)
                                    await self._raise_tool_call_delta(collected_tool_calls, **callback_opts)
                                else:
                                    await self._raise_system_event(f"content_block_start Unknown content type: {content_type}", callback_opts)
                            elif event.type == "input_json":
                                collected_tool_calls[-1]['input'] = event.snapshot
                            elif event.type == "text":
                                collected_messages.append(event.text)
                                await self._raise_text_delta(event.text, **callback_opts)
                            elif event.type == 'message_delta':
                                stop_reason = event.delta.stop_reason


                except APITimeoutError as e:
                    await self._raise_system_event(f"Timeout error calling `client.messages.stream`. Delaying for {delay} seconds.\n", **callback_opts)
                    await self._exponential_backoff(delay)
                    delay *= 2
                except Exception as e:
                    await self._raise_system_event(f"Exception calling `client.messages.create`.\n\n{e}\n", **callback_opts)
                    await self._raise_completion_end(opts["completion_opts"], stop_reason="exception", **callback_opts)

                    return []

        return messages

    def _generate_multi_modal_user_message(self, user_input: str, images: List[ImageInput], audio: List[AudioInput]) -> Union[List[dict[str, Any]], None]:
        contents = []
        for image in images:
            if image.content is None and image.url is not None:
                logging.warning(f"ImageInput has no content and Claude doesn't support image URLs. Skipping image {image.url}")
                continue

            img_source = {"type": "base64", "media_type": image.content_type, "data": image.content}
            contents.append({"type": "image", "source": img_source})

        contents.append({"type": "text", "text": user_input})

        return [{"role": "user", "content": contents}]

    async def one_shot(self, **kwargs) -> str:
        """
               Perform the chat operation as a one shot.

               Parameters:
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

    async def __tool_calls_to_messages(self, tool_calls, tool_chest):
        async def make_call(tool_call):
            fn = tool_call['name']
            args = tool_call['input']
            ai_call = copy.deepcopy(tool_call)
            try:
                function_response = await self._call_function(tool_chest, fn, args)
                call_resp = {"type": "tool_result", "tool_use_id": tool_call['id'],"content": function_response}
            except Exception as e:
                call_resp = {"role": "tool", "tool_call_id": tool_call['id'], "name": fn,
                             "content": f"Exception: {e}"}

            return ai_call, call_resp

        # Schedule all the calls concurrently
        tasks = [make_call(tool_call) for tool_call in tool_calls]
        completed_calls = await asyncio.gather(*tasks)

        # Unpack the resulting ai_calls and resp_calls
        ai_calls, results = zip(*completed_calls)

        return [{'role': 'assistant', 'content': list(ai_calls)},
                {'role': 'user', 'content': list(results)}]