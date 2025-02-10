from queue import Queue

from agent_c.agents.base import BaseAgent


class AgentInterface:
    def __init__(self, agent_obj: BaseAgent, queue_cls = Queue):
        self.agent_obj = agent_obj
        self.input_queue = queue_cls()

