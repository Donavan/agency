import base64
from collections import defaultdict

import asyncio
import logging
import uuid
from typing import Dict, Optional, List, Any, AsyncGenerator

from agent_c import BaseAgent, Toolset, ChatEvent
from agent_c_reference_apps.react_fastapi.backend.backend_app.reactjs_agent import ReactJSAgent
from agent_c_reference_apps.react_fastapi.backend.util.logging_utils import LoggingManager


# logging.basicConfig(
#     level=logging.DEBUG,
#     encoding='utf-8',
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.StreamHandler(),
#         logging.FileHandler('agent_manager.log'),
#     ]
# )
# my_logger = logging.getLogger("httpx")
# my_logger.setLevel(logging.ERROR)


class AgentManager:
    """
    Manages agent sessions in a multi-agent environment.

    This class handles the lifecycle of agent sessions, including creation,
    cleanup, and response streaming. It supports different LLM backends and
    tool configurations while ensuring thread-safe operations through
    per-session locks.

    Attributes:
        ESSENTIAL_TOOLS (List[str]): Required tool sets for all agents
        logger (logging.Logger): Instance logger
        sessions (Dict[str, Dict[str, Any]]): Active session storage
        _locks (Dict[str, asyncio.Lock]): Session operation locks
    """
    ESSENTIAL_TOOLS = ['MemoryTools', 'WorkspaceTools', 'PreferenceTools', 'RandomNumberTools']

    def __init__(self):
        logging_manager = LoggingManager(__name__)
        self.logger = logging_manager.get_logger()
        # self.debug_event = LoggingManager.get_debug_event()

        self.sessions: Dict[str, Dict[str, Any]] = {}
        self._locks: Dict[str, asyncio.Lock] = {}


    def get_session_data(self, session_id: str) -> Dict[str, Any]:
        """
        Retrieve session data for a given session ID.

        Args:
            session_id (str): The unique identifier for the session

        Returns:
            Dict[str, Any]: Session data dictionary containing agent and configuration
                information, or None if session doesn't exist
        """
        return self.sessions.get(session_id)

    async def create_session(self, llm_model: str = "gpt-4o", backend: str = 'openai',
                             persona_name: str = 'default',
                             additional_tools: List[str] = None,
                             existing_session_id: str = None, **kwargs) -> str:
        """
        Create a new session or update an existing session with a new agent.

        Args:
            llm_model: The model to use
            backend: The backend provider ('openai' or 'claude')
            persona_name: The name of the persona to use
            existing_session_id: If provided, updates existing session instead of creating new one
            additional_tools: Additional tools to add to the agent - string names of tool class names
            **kwargs: Additional keyword arguments: currently passes kwargs to ReactJSAgent

        Returns:
            str: The session ID (either new or existing)

        Raises:
            Exception: If agent initialization fails
        """
        # If updating existing session, use that ID, otherwise generate new one
        session_id = existing_session_id if existing_session_id else str(uuid.uuid4())

        # Create lock if it doesn't exist
        if session_id not in self._locks:
            self._locks[session_id] = asyncio.Lock()

        async with self._locks[session_id]:
            # Get existing session data if updating
            existing_session = self.sessions.get(session_id, {})
            existing_agent: BaseAgent | None = existing_session.get("agent", None)

            # Initialize agent for this session
            agent = ReactJSAgent(
                user_id=session_id,
                backend=backend,
                model_name=llm_model,
                additional_tools=additional_tools or [],
                persona_name=persona_name,
                agent_name=f"Agent_{session_id}",
                **kwargs
            )

            # If updating existing session, transfer necessary state
            if existing_agent:
                # Transfer session manager to maintain history - I need to pass this first
                agent.session_manager = existing_agent.session_manager

            # Initialize the agent
            await agent.initialize()

            # Transfer any other necessary state
            if existing_agent:
                agent.custom_persona_text = existing_agent.custom_persona_text

            # Update sessions dictionary
            self.sessions[session_id] = {
                "agent": agent,
                "llm_model": llm_model,
                "created_at": agent._current_timestamp(),
                "agent_name": f"Agent_{session_id}",
                "agent_c_session_id": agent.session_id,
            }

            return session_id

    async def cleanup_session(self, session_id: str):
        """
        Clean up resources associated with a specific session.

        Removes session data and releases associated locks. Any errors during
        cleanup are logged but don't halt execution.

        Args:
            session_id (str): The unique identifier of the session to clean up

        Note:
            This method is safe to call multiple times on the same session ID
        """
        if session_id in self.sessions:
            try:
                session_data = self.sessions[session_id]
                # agent: BaseAgent = session_data.get("agent")
                # if agent:
                # if hasattr(agent, 'tool_chest') and agent.tool_chest:
                #     await agent.tool_chest.cleanup()
                # if hasattr(agent, 'session_manager') and agent.session_manager:
                #     await agent.session_manager.close()

                # Remove session data and lock
                del self.sessions[session_id]
                if session_id in self._locks:
                    del self._locks[session_id]

            except Exception as e:
                self.logger.error(f"Error cleaning up session {session_id}: {e}")

    async def stream_response(
            self,
            session_id: str,
            user_message: str,
            custom_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Get streaming response from the agent for a given message.
        Uses ReactJSAgent's stream_chat method for proper streaming support.

        Args:
            session_id: The session identifier
            user_message: The user's message to process
            custom_prompt: Optional custom prompt to use

        Yields:
            Chunks of the response as they become available

        Raises:
            ValueError: If the session ID is invalid
            Exception: If streaming fails
        """
        session_data = self.get_session_data(session_id)
        if not session_data:
            raise ValueError(f"Invalid session ID: {session_id}")

        agent = session_data["agent"]

        try:
            # Use the new streaming method
            async for chunk in agent.stream_chat(
                    user_message=user_message,
                    custom_prompt=custom_prompt
            ):
                yield chunk

        except Exception as e:
            self.logger.error(f"Error in stream_response: {e}")
            yield f"Error: {str(e)}"
