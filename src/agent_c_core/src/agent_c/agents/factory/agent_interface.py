import asyncio
import logging

from typing import Optional

from asyncio import Queue as AsyncQueue
from agent_c.agents.base import BaseAgent


class AgentInterface:
    def __init__(self, agent_obj: BaseAgent, queue_cls = AsyncQueue, logger=None):
        self.agent_obj = agent_obj
        self.input_queue = queue_cls()
        self.output_queue = queue_cls()
        self._running = False
        self._process_task: Optional[asyncio.Task] = None
        self.logger = logging.getLogger(__name__)

    async def start(self):
        """Start processing queues"""
        if self._running:
            return

        self._running = True
        self._process_task = asyncio.create_task(self._process_queue())
        self.logger.info("Agent interface queue processing started")

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



    async def _handle_input_message(self, input_message):
        pass

    async def _process_queue(self):
        """Main processing loop for handling queue messages"""
        while self._running:
            if not self.input_queue.empty():
                try:

                    input_message = await self.input_queue.get()
                except asyncio.CancelledError:
                    break

                if input_message is None:
                    break

                await self._handle_input_message(input_message)
            else:
                await asyncio.sleep(0.01)

