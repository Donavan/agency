import asyncio
import logging

from typing import Optional
from asyncio import Queue as AsyncQueue

from torch import meshgrid

from agent_c import BaseModel
from agent_c.agents.base import BaseAgent
from agent_c.util.slugs import MnemonicSlugs
from agent_c.agents.factory.agent_factory import AgentFactory
from agent_c.models.events.Interaction_error import InteractionErrorEvent
from agent_c.models.agent_interface.backends import AvailableBackendsResponse
from agent_c.models.agent_factory_request import AgentCreationOptions, AgentRuntimeParams
from agent_c.models.interaction.requests.interaction_start import InteractionStartRequest

class ActiveAgentInteraction(BaseModel):
    interaction_id: str
    session_id: str
    agent: BaseAgent
    agent_input_queue: AsyncQueue = AsyncQueue()
    agent_output_queue: AsyncQueue = AsyncQueue()
    messages: list = []  # This will be in out format
    want_cancel: bool = False

    @property
    def full_id(self):
        return f"{self.session_id}.{self.interaction_id}"

class AgentInterface:
    def __init__(self, agent_factory: AgentFactory, queue_cls = AsyncQueue, logger=None):
        self._agent_cache: dict[str, BaseAgent] = {}
        self._active_interactions: dict[str, ActiveAgentInteraction] = {}
        self.factory = agent_factory
        self.input_queue = queue_cls()
        self.output_queue = queue_cls()

        self._running = False
        self._process_task: Optional[asyncio.Task] = None
        self.logger = logger or logging.getLogger(__name__)

    async def start(self):
        """Start processing queues"""
        if self._running:
            return

        self._running = True
        self._process_task = asyncio.create_task(self._process_queue())
        self.logger.info("Agent interface queue processing started")
        await self.output_queue.put(AvailableBackendsResponse(backends=self.factory.available_backends))

    async def stop(self):
        """Stop processing queues"""
        if not self._running:
            return

        self._running = False
        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Agent interface queue processing stopped")

    async def _chat_callback(self, event):
        await self.output_queue.put(event)

    async def _handle_interaction_agent_output(self, event, interaction: ActiveAgentInteraction):
        pass

    async def _service_interaction(self, interaction: ActiveAgentInteraction):
        while True:
            if interaction.want_cancel:
                break

            if not interaction.agent_output_queue.empty():
                try:
                    while not interaction.agent_output_queue.empty():
                        await self._handle_interaction_agent_output(await interaction.agent_ouput_queue.get(),
                                                                    interaction)
                except asyncio.CancelledError:
                    break
            else:
                await asyncio.sleep(0.01)

    async def _initiate_interaction(self, request: InteractionStartRequest):
        # TODO: 1. grab the session history from the session manager
        #       2. construct the message array in native format
        #       3. kick off the agent interaction
        #       4. start the service_interaction loop
        #
        agent = self._agent_cache[request.interaction_id]
        interaction = ActiveAgentInteraction(interaction_id=request.interaction_id, agent=agent)
        self._active_interactions[request.interaction_id] = interaction

        asyncio.create_task(self._service_interaction(interaction))

    async def _interaction_start(self, request: InteractionStartRequest):
        if request.backend not in self.factory.available_backends:
            self.logger.error(f"Backend {request.backend} is not available")
            await self.output_queue.put(InteractionErrorEvent(interaction_id=request.interaction_id,
                                                              error_message=f"Backend {request.backend} is not available",
                                                              session_id=request.session_id))
            return

        if request.interaction_id is None:
            request.interaction_id = MnemonicSlugs.generate_slug(3)

        if request.backend not in self._agent_cache:
            agent = self.factory.create_agent_runtime(AgentCreationOptions(runtime=AgentRuntimeParams(backend=request.backend)))
            self._agent_cache[request.backend] = agent

        asyncio.create_task(self._initiate_interaction(request))






    async def _handle_input_message(self, input_message):
        if input_message.type == "interaction_start":
            await self._interaction_start(input_message)
        elif input_message.type == "available_backends_request":
            await self.output_queue.put(AvailableBackendsResponse(backends=self.factory.available_backends))

    async def _process_queue(self):
        """Main processing loop for handling queue messages"""
        while self._running:
            if not self.input_queue.empty():
                try:
                    input_message = await self.input_queue.get()
                except asyncio.CancelledError:
                    break

                if input_message is not None:
                    asyncio.create_task(self._handle_input_message(input_message))
            else:
                await asyncio.sleep(0.01)

