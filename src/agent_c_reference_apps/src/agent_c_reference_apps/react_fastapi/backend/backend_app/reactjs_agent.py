import base64
from collections import defaultdict

import asyncio
import json
import logging
import os

import threading
from typing import Union, List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime, timezone

from agent_c import DynamicPersonaSection, ChatEvent
from agent_c.agents import GPTChatAgent
from agent_c.agents.claude import ClaudeChatAgent
from agent_c.models.events import SessionEvent
from agent_c_reference_apps.react_fastapi.backend.util.logging_utils import LoggingManager
from agent_c_tools.tools.workspaces import LocalStorageWorkspace
from agent_c.util import debugger_is_active
from agent_c.toolsets import ToolChest, ToolCache, Toolset

from agent_c.chat import ChatSessionManager

from agent_c_tools.tools.user_preferences import AssistantPersonalityPreference, AddressMeAsPreference, UserPreference
from agent_c.prompting import PromptBuilder, CoreInstructionSection, HelpfulInfoStartSection, \
    EndOperatingGuideLinesSection, \
    EnvironmentInfoSection

from agent_c_tools.tools.user_bio.prompt import UserBioSection


class ReactJSAgent:
    """
    A bridge interface between the agent_c library and ReactJS applications for chat functionality.

    This class provides a comprehensive interface for managing AI chat interactions,
    including session management, tool integration, and streaming responses. It supports
    multiple AI backends (OpenAI, Claude) and provides dynamic tool loading capabilities.

    Features:
        - Asynchronous chat streaming
        - Dynamic tool management
        - Multiple AI backend support
        - Custom persona management
        - Comprehensive event handling
        - Workspace management
    """

    def __init__(self, user_id: str = 'default', session_manager: Union[ChatSessionManager, None] = None,
                 backend: str = 'openai', model_name: str = 'gpt-4o', persona_name: str = 'default',
                 custom_persona_text: str = '',
                 essential_tools: List[str] = None,
                 additional_tools: List[str] = None,
                 **kwargs):
        """
        Initialize the SimplifiedAgent instance.

        Args:
            user_id (str): Identifier for the user. Defaults to 'default'.
            session_manager (Union[ChatSessionManager, None]): Session manager for chat sessions. Defaults to None.
            backend (str): Backend to use for the agent (e.g., 'openai', 'claude'). Defaults to 'openai'.
            model_name (str): Name of the AI model to use. Defaults to 'gpt-4o'.
            additional_tools (List[str]): List of additional tools to add to the agent. Defaults to None.
            **kwargs: Additional optional keyword arguments.
        """


        # Agent events setup, must come first
        self.__init_events()

        # Debugging and Logging Setup
        logging_manager = LoggingManager(__name__)
        self.logger = logging_manager.get_logger()
        self.debug_event = None

        self.agent_name = kwargs.get('agent_name', None)  # Debugging only

        # Initialize Core Class, there's quite a bit here, so let's go through this.
        # Agent Characteristics
        # - Backend: The backend used for the agent, defaults to 'openai', other open is 'claude'
        # - Model Name: The model name used for the agent, defaults to 'gpt-4o'
        self.backend = backend
        self.model_name = model_name
        self.agent = None
        self.agent_output_format = kwargs.get('output_format', 'raw')
        self.temperature = kwargs.get('temperature', 0.5)
        self.reasoning_effort = kwargs.get('reasoning_effort', 'medium')

        # Agent "User" Management - these are used to keep agents and sessions separate in zep cache
        # - User ID: The user ID for the agent, this is used for user preference management, it is required.
        # In the future, I'll probably change this to a UUID
        # - User Preferences: A list of user preferences that the agent can use, this is a list of UserPreference objects.
        self.user_id = user_id
        self.user_prefs: List[UserPreference] = [AddressMeAsPreference(), AssistantPersonalityPreference()]
        self.session_manager = session_manager

        # Agent persona management, pass in the persona name and we'll initialize the persona text now
        # we will pass in custom text later
        self.persona_name = persona_name
        self.custom_persona_text = custom_persona_text
        if self.persona_name is None or self.persona_name == '':
            self.persona_name = 'default'
        try:
            self.custom_persona_text = self.__load_persona(self.persona_name)
            self.logger.info(f"Loaded persona: {self.persona_name} for user_id: {self.user_id}")
        except Exception as e:
            self.logger.error(f"Error loading persona {self.persona_name}: {e}")

        # Chat Management, this is where the agent stores the chat history
        self.current_chat_Log: Union[List[Dict], None] = None

        # Tool Chest, Cache, and Setup
        # - Tool Chest: A collection of toolsets that the agent can use, Set to None initialization.
        # - Tool Cache: A cache for storing tool data, set to a default directory.
        # - Additional Tools: A list of additional toolsets to add to the tool chest, this is where use passes in other toolsets
        # - Selected Tools: A list of toolsets that the agent will use, this is a combination of essential toolsets and additional toolsets.
        # - Output Tool Arguments: A placeholder for tool argument output preference.
        self.tool_chest: Union[ToolChest, None] = None
        self.tool_cache_dir = kwargs.get("tool_cache_dir", ".tool_cache")
        self.tool_cache = ToolCache(cache_dir=self.tool_cache_dir)

        if essential_tools is None:
            self.essential_tools = ['MemoryTools', 'WorkspaceTools', 'PreferenceTools', 'RandomNumberTools']
        else:
            self.essential_tools = essential_tools
        self.additional_tools = additional_tools or []
        self.selected_tools = self.essential_tools + self.additional_tools
        self.output_tool_arguments = True  # Placeholder for tool argument output preference

        # Agent Workspace Setup
        self.workspaces = None
        self.__init_workspaces()

        self.voice_tools = None

    def __init_events(self):
        """
        Initialize threading events used for debugging and input/output management.
        """
        self.exit_event = threading.Event()
        self.input_active_event = threading.Event()
        self.cancel_tts_event = threading.Event()
        # going to get from shared logging manager
        self.debug_event = LoggingManager.get_debug_event()


    def __load_persona(self, persona_name: str = None) -> str:
        """
        Load the persona prompt from a file based on the given persona name.

        Returns:
            str: Loaded persona prompt text.

        Raises:
            Exception: If the persona file cannot be loaded.
        """
        if persona_name is None or persona_name == '' or persona_name.lower() == 'custom':
            return ''
        try:
            self.logger.info(f"Agent {self.agent_name} is loading persona: {self.persona_name}")
            persona_path = os.path.join('personas', f"{self.persona_name}.md")

            # If persona file exists, read it
            if os.path.exists(persona_path):
                with open(persona_path, 'r') as file:
                    return file.read()

            # If no persona file, return empty string
            self.logger.warning(f"No persona file found for {self.persona_name}. ")
            return ''

        except Exception as e:
            self.logger.error(f"Error loading persona {self.persona_name}: {e}")
            return ''

    def __init_workspaces(self):
        """
        Initialize the agent's workspaces by loading local workspace configurations.
        """
        self.workspaces = [LocalStorageWorkspace(name="project", workspace_path=os.getcwd(),
                                                 description="A workspace holding the `Agent C` source code in Python.")]

        try:
            local_workspaces = json.load(open(".local_workspaces.json", "r"))
            for ws in local_workspaces['local_workspaces']:
                self.workspaces.append(LocalStorageWorkspace(**ws))
        except FileNotFoundError:
            pass

    async def __init_session(self):
        """
        Initialize the chat session for the agent, including setting up the session manager.
        """
        if self.session_manager is None:
            self.session_manager = ChatSessionManager()
            await self.session_manager.init(self.user_id)

        self.session_id = self.session_manager.chat_session.session_id
        print(f"###Agent {self.agent_name} has initialized with session: {self.session_id}")

    async def update_tools(self, new_tools: List[str]):
        """
        Updates the agent's tools without reinitializing the entire agent.

        This method allows dynamic updating of the tool set while maintaining
        the current session and other configurations. It ensures essential
        tools are preserved while adding or removing additional tools.

        Args:
            new_tools (List[str]): List of tool names to be added to the essential tools

        Notes:
            - Essential tools are always preserved
            - Duplicate tools are automatically removed
            - Tool chest is reinitialized with the updated tool set
            - Agent is reinitialized with new tools while maintaining the session
        """
        print(f"Updating tools for agent {self.agent_name} to: {new_tools}")

        # Remove duplicates and ensure essential tools are included
        all_tools = list(set(self.essential_tools + new_tools))

        self.additional_tools = [t for t in new_tools if t not in self.essential_tools]
        self.selected_tools = all_tools

        # Reinitialize just the tool chest
        await self.__init_tool_chest()

        # Reinitialize the agent with new tools but keep the session
        await self.__init_agent()

        print(f"Tools updated successfully. Active tools: {list(self.tool_chest.active_tools.keys())}")

    async def __init_tool_chest(self):
        """
        Initialize the agent's tool chest with selected tools and configurations.

        This method sets up the `ToolChest` with tools from the global `Toolset` registry
        based on the tools specified in `self.selected_tools`. It also configures additional
        tool options, initializes the selected tools, and logs the result. The method handles
        errors and logs any tools that failed to initialize, providing useful debugging information.

        Steps:
        1. Filter tools from the `Toolset` registry to match those in `self.selected_tools`.
        2. Initialize the `ToolChest` with the filtered tools.
        3. Create and set up a `ToolCache` for caching tool-related data.
        4. Pass the necessary configuration options to initialize the tools in the `ToolChest`.
        5. Log the successfully initialized tools.
        6. Debug and log any tools that were selected but failed to initialize.

        Raises:
            Exception: Logs and prints an error message if the tool initialization fails.

        Attributes:
            self.tool_chest (ToolChest): The tool chest instance containing the initialized tools.
            self.tool_cache (ToolCache): Cache manager for tool-related data.
            self.selected_tools (list): Names of tools selected for initialization.
            self.logger (Logger): Logger instance for logging messages.
            self.agent_name (str): The name of the agent using this tool chest.

        Debugging:
            Logs warnings if selected tools do not get initialized, typically due to
            misspelled tool class names in the `self.selected_tools` list.

        Returns:
            None
        """
        self.logger.info(f"Selected tools: {self.selected_tools}")

        self.tool_chest = ToolChest(tool_classes=[
            tool for tool in Toolset.tool_registry
            if tool.__name__ in self.selected_tools
        ])

        try:
            self.tool_cache = ToolCache(cache_dir=".tool_cache")
            tool_opts = {
                'tool_cache': self.tool_cache,
                'session_manager': self.session_manager,
                'user_preferences': self.user_prefs,
                'workspaces': self.workspaces,
                'streaming_callback': self.consolidated_streaming_callback
            }
            await self.tool_chest.init_tools(**tool_opts)
            self.logger.info(
                f"Agent {self.agent_name} has initialized tools: {list(self.tool_chest.active_tools.keys())}")

            # From this point on, this is only additional debugging code to troubleshoot when tools don't get initialized
            # Usually it's a misspelling of the tool class name from the LLM
            initialized_tools = set(self.tool_chest.active_tools.keys())
            tool_class_to_instance_name = {
                tool.__name__: instance.name
                for instance in self.tool_chest.active_tools.values()
                for tool in Toolset.tool_registry
                if isinstance(instance, tool)
            }
            # Find tools that were selected but not initialized
            uninitialized_tools = [
                tool_name for tool_name in self.selected_tools
                if tool_class_to_instance_name.get(tool_name) not in initialized_tools
            ]
            if uninitialized_tools:
                self.logger.warning(
                    f"The following selected tools were not initialized: {uninitialized_tools} for Agent {self.agent_name}")
        except Exception as e:
            print(f"Error initializing tools: {e}")
        return

    async def __init_agent(self):
        """
        Initialize the internal agent with prompt builders, tools, and configurations.

        This method creates and configures the conversational agent for the application.
        It sets up the agent's operational and informational sections, constructs a prompt
        builder with these sections, and initializes the agent based on the specified backend.

        Steps:
        1. Define the operational sections of the agent, including core instructions, dynamic
           persona settings, and active tool-related sections from the `ToolChest`.
        2. Define the informational sections, such as environmental context and user bio details.
        3. Combine operational and informational sections into a `PromptBuilder` instance.
        4. Initialize the agent using the `ClaudeChatAgent` or `GPTChatAgent` class,
           depending on the specified backend (`self.backend`).

        Attributes:
            self (ChatAgent): The initialized conversational agent instance
                (either `ClaudeChatAgent` or `GPTChatAgent`).
            self.backend (str): The backend type, determining which agent class is used.
            self.model_name (str): The name of the AI model used by the agent.
            self.tool_chest (ToolChest): The initialized tool chest containing active tools.
            self.__chat_callback (function): Callback function for handling streaming responses.
            self.agent_output_format (str): Format for the agent's output.

        Raises:
            Exception: If an error occurs during agent initialization, an exception may be raised.

        Returns:
            None
        """
        operating_sections = [
            CoreInstructionSection(template="<operating_guidelines>\n"),
            DynamicPersonaSection(),
            EndOperatingGuideLinesSection()
        ]
        operating_sections.extend(self.tool_chest.active_tool_sections)

        info_sections = [
            HelpfulInfoStartSection(),
            EnvironmentInfoSection(session_manager=self.session_manager, voice_tools=None, agent_voice=None),
            UserBioSection(session_manager=self.session_manager)
        ]

        prompt_builder = PromptBuilder(sections=operating_sections + info_sections)

        if self.backend == 'claude':
            self.agent = ClaudeChatAgent(
                prompt_builder=prompt_builder,
                model_name=self.model_name,
                tool_chest=self.tool_chest,
                streaming_callback=self.consolidated_streaming_callback,
                output_format=self.agent_output_format
            )
        else:
            self.agent = GPTChatAgent(
                prompt_builder=prompt_builder,
                model_name=self.model_name,
                tool_chest=self.tool_chest,
                streaming_callback=self.consolidated_streaming_callback,
                output_format=self.agent_output_format
            )

    async def __build_prompt_metadata(self) -> Dict[str, Any]:
        """
        Build metadata for prompts including user and session information.

        Returns:
            Dict[str, Any]: Metadata for prompts.
            - session_id (str): Session ID for the chat session.
            - current_user_username (str): Username of the current user.
            - current_user_name (str): Name of the current user.
            - session_summary (str): Summary of the current chat session.
            - persona_prompt (str): Prompt for the persona.
        """
        return {
            "session_id": self.session_id,
            "current_user_username": self.session_manager.user.user_id,
            "current_user_name": self.session_manager.user.first_name,
            "session_summary": self.session_manager.chat_session.active_memory.summary,
            "persona_prompt": self.custom_persona_text,
            "voice_tools": self.voice_tools,  # Add voice tools to metadata
            "timestamp": datetime.now().isoformat(),
            "env_name": os.getenv('ENV_NAME', 'development'),
            "session_info": self.session_manager.chat_session.session_id if self.session_manager else None
        }

    @staticmethod
    def _current_timestamp() -> str:
        """
        Returns the current UTC timestamp as a string.

        Returns:
            str: Current timestamp in ISO format.
        """
        return datetime.now(timezone.utc).isoformat()

    def _get_agent_config(self) -> Dict[str, Any]:
        """
        Get the current configuration of the agent.

        Returns:
            Dict[str, Any]: Dictionary containing agent configuration details
        """
        config = {
            'user_id': self.user_id,
            'backend': self.backend,
            'model_name': self.model_name,
            'persona_name': self.persona_name,
            'initialized_tools': [],
            'agent_name': self.agent_name,
            'output_format': self.agent_output_format,
            'created_time': self._current_timestamp(),
            'temperature': self.temperature,
            'reasoning_effort': self.reasoning_effort,
            'model_parameters': {
                'temperature': getattr(self, 'temperature', None),
                'reasoning_effort': getattr(self, 'reasoning_effort', None)
            }
        }

        if self.tool_chest:
            config['initialized_tools'] = [{
                'instance_name': instance_name,
                'class_name': tool_instance.__class__.__name__,
                'developer_tool_name': getattr(tool_instance, 'name', instance_name),
                'description': tool_instance.__class__.__doc__
            } for instance_name, tool_instance in self.tool_chest.active_tools.items()]

        self.logger.debug(f"Agent {self.agent_name} reporting config: {config}")
        return config

    @staticmethod
    async def _handle_text_delta(event):
        """Handle text delta events from the agent/tools"""
        payload = json.dumps({
            "type": "content",
            "data": event.content,
            "format": event.format
        }) + "\n"
        return payload

    @staticmethod
    async def _handle_tool_call(event):
        """
        Unified tool call handler that checks the vendor to determine how to send messages.
        The front end expects a schema: tool_calls has id, name, arguments.  tool_results has role, tool_call_id, name, and content.
        For OpenAI:
          - When event.active is True, send a "tool_calls" message.
          - When event.active is False, send a "tool_results" message.

        For Claude:
          - Since Claude may not send an active tool call message, always emit both a
            "tool_calls" and a "tool_results" message (with active set to False) when complete.
        """
        # Ensure tool_calls is always a list.
        if event.tool_calls is None:
            calls = []
        elif isinstance(event.tool_calls, list):
            calls = event.tool_calls
        else:
            calls = [event.tool_calls]

        # Ensure tool_results is always a list.
        if event.tool_results is None:
            results = []
        elif isinstance(event.tool_results, list):
            results = event.tool_results
        else:
            results = [event.tool_results]

        if event.vendor == "open_ai":
            # For OpenAI, follow the standard streaming behavior.
            if event.active:
                payload_obj = {
                    "type": "tool_calls",
                    "tool_calls": calls
                }
            else:
                payload_obj = {
                    "type": "tool_results",
                    "tool_results": results
                }
            return json.dumps(payload_obj) + "\n"
        elif event.vendor == "anthropic":
            # For Claude, send tool calls if we have calls, and tool results if we have results
            if event.tool_results:  # We have results
                tool_results = []
                for result in results:
                    payload_result = {
                        "role": "tool",
                        "tool_call_id": result.get("tool_use_id", ""),
                        "name": calls[0].get("name", "") if calls else "",
                        "content": result.get("content", "{}")
                    }
                    tool_results.append(payload_result)

                return json.dumps({
                    "type": "tool_results",
                    "tool_results": tool_results
                }) + "\n"
            else:  # This is just a tool call
                tool_calls = []
                for call in calls:
                    tool_call = {
                        "id": call.get("id", ""),
                        "name": call.get("name", ""),
                        "arguments": json.dumps(call.get("input", {}))
                    }
                    tool_calls.append(tool_call)

                return json.dumps({
                    "type": "tool_calls",
                    "tool_calls": tool_calls
                }) + "\n"
        else:
            raise ValueError(f"Unsupported vendor: {event.vendor}")

    @staticmethod
    async def _handle_render_media(event):
        """Handle media render events from tools"""
        media_content = ""
        if event.content:
            media_content = event.content
        elif event.url:
            if "image" in event.content_type:
                media_content = f"<br><img src='{event.url}' style='max-width: 60%; height: auto;'/>"
            else:
                media_content = f"<br><object type='{event.content_type}' data='{event.url}'></object>"
        payload = json.dumps({
            "type": "render_media",
            "content": media_content,
            "content_type": event.content_type,
            "role": event.role,
            "metadata": {
                "sent_by_class": getattr(event, 'sent_by_class', None),
                "sent_by_function": getattr(event, 'sent_by_function', None),
                "session_id": getattr(event, 'session_id', None),
                "name": getattr(event, 'name', None)
            }
        }) + "\n"
        return payload

    @staticmethod
    async def _handle_history(event):
        """Handle history events which update the chat log"""
        # self.current_chat_Log = event.messages
        payload = json.dumps({
            "type": "history",
            "messages": event.messages
        }) + "\n"
        return payload

    @staticmethod
    async def _handle_audio_delta(event):
        """Handle audio events if voice features are enabled"""
        # For tools/agents these typically don't need special handling
        return None

    @staticmethod
    async def _handle_completion(event):
        """Handle completion events from the agent"""
        payload = json.dumps({
            "type": "completion_status",
            "data": {
                "running": event.running,
                "stop_reason": event.stop_reason,
                "input_tokens": event.input_tokens,
                "output_tokens": event.output_tokens,
            }
        }) + "\n"
        return payload

    @staticmethod
    async def _handle_interaction(event):
        """Handle interaction state events"""
        if event.started:
            payload = json.dumps({
                "type": "interaction_start",
                "id": event.id,
                "session_id": event.session_id
            }) + "\n"
        else:
            payload = json.dumps({
                "type": "interaction_end",
                "id": event.id,
                "session_id": event.session_id
            }) + "\n"
        return payload

    async def initialize(self):
        """
        Asynchronously initialize the agent's session, tool chest, and internal agent configuration.
        """
        await self.__init_session()
        await self.__init_tool_chest()
        await self.__init_agent()

    async def consolidated_streaming_callback(self, event: SessionEvent):
        """
        Processes and routes various types of session events through appropriate handlers.

        This method serves as the central event processing hub, handling various event
        types including text updates, tool calls, media rendering, and completion status.
        It formats the events into JSON payloads suitable for streaming to the client.

        Args:
            event (SessionEvent): The event to process, containing type and payload information

        Event Types Handled:
            - text_delta: Text content updates
            - tool_call: Tool invocation events
            - render_media: Media content rendering
            - history: Chat history updates
            - audio_delta: Audio content updates
            - completion: Task completion status
            - interaction: Interaction state changes

        Notes:
            Events are processed and formatted into JSON strings with appropriate
            type markers and payloads for client-side handling.
        """
        try:
            self.logger.debug(
                f"Consolidated callback received event: {event.model_dump_json(exclude={'content_bytes'})}")
        except Exception as e:
            self.logger.debug(f"Error serializing event {event.type}: {e}")

        # A simple dispatch dictionary that maps event types to handler methods.
        handlers = {
            "text_delta": self._handle_text_delta,
            "tool_call": self._handle_tool_call,
            "render_media": self._handle_render_media,
            "history": self._handle_history,
            "audio_delta": self._handle_audio_delta,
            "completion": self._handle_completion,
            "interaction": self._handle_interaction,
        }

        handler = handlers.get(event.type)
        if handler:
            # Each handler method is responsible for formatting the payload.
            payload = await handler(event)
            if payload is not None and hasattr(self, "_stream_queue"):
                await self._stream_queue.put(payload)

            # If this is the end-of-stream event, push a termination marker.
            # Client must handle a None payload to know the stream has ended.
            if event.type == "interaction" and not event.started:
                # Push a None to signal the end of the stream.
                await self._stream_queue.put(None)
        else:
            self.logger.warning(f"Unhandled event type: {event.type}")

    async def stream_chat(self, user_message: str, custom_prompt: str = None) -> AsyncGenerator[str, None]:
        """
        Streams chat responses for a given user message.

        This method handles the complete chat interaction process, including:
        - Session updates
        - Custom prompt integration
        - Message processing
        - Response streaming
        - Event handling

        Args:
            user_message (str): The message from the user to process
            custom_prompt (str, optional): Custom prompt to override default persona. Defaults to None.

        Yields:
            str: JSON-formatted strings containing various response types:
                - Content updates ("type": "content")
                - Tool calls ("type": "tool_calls")
                - Tool results ("type": "tool_results")
                - Media renders ("type": "render_media")
                - Completion status ("type": "completion_status")
                - Errors ("type": "error")

        Raises:
            Exception: Any errors during chat processing
        """
        queue = asyncio.Queue()
        self._stream_queue = queue  # Make the queue available to the callback

        try:
            await self.session_manager.update()

            if custom_prompt is not None:
                self.custom_persona_text = custom_prompt

            # Set the agent’s streaming callback to our consolidated version.
            original_callback = self.agent.streaming_callback
            self.agent.streaming_callback = self.consolidated_streaming_callback

            prompt_metadata = await self.__build_prompt_metadata()

            chat_task = asyncio.create_task(
                self.agent.chat(
                    streaming_queue=queue,
                    session_manager=self.session_manager,
                    user_message=user_message,
                    prompt_metadata=prompt_metadata,
                    messages=self.current_chat_Log,
                    output_format='raw',
                    temperature=self.temperature
                )
            )

            while True:
                try:
                    content = await queue.get()
                    if content is None:
                        break
                    yield content
                    queue.task_done()
                except asyncio.CancelledError:
                    self.logger.info("Stream was cancelled")
                    break

            await chat_task
            # Restore the original callback
            self.agent.streaming_callback = original_callback
            await self.session_manager.flush()

        except Exception as e:
            self.logger.error(f"Error in stream_chat: {e}")
            yield json.dumps({
                "type": "error",
                "data": str(e)
            }) + "\n"
        finally:
            # Drain any remaining items in the queue.
            while not queue.empty():
                try:
                    await queue.get_nowait()
                    queue.task_done()
                except asyncio.QueueEmpty:
                    break
