from agent_c.agents.claude import ClaudeChatAgent
from agent_c.agents.gpt import GPTChatAgent
from agent_c.models.agent_factory_request import AgentFactoryRequest

class AgentFactory:
    __backend_to_agent_map = {
        "gpt": GPTChatAgent,
        "claude": ClaudeChatAgent
    }

    def __init__(self, **kwargs):
        backends = kwargs.get('backends', ['gpt', 'claude'])
        self._backend_client_map = {}

        for backend in backends:
            try:
                backend_cls = self.__backend_to_agent_map[backend]
                self._backend_client_map[backend] = kwargs.get(f"{backend}_client", backend_cls.default_client())
            except Exception as e:
                continue

        if len(self._backend_client_map) == 0:
            raise ValueError("No valid backends available")

    def create_agent(self, request: AgentFactoryRequest):
        if request.backend not in self.__backend_to_agent_map:
            raise ValueError(f"Invalid backend  {request.backend}")
        elif request.backend not in self._backend_client_map:
            raise ValueError(f"No client available for backend {request.backend}")

        agent_cls = self.__backend_to_agent_map.get(request.backend)
        agent_obj = agent_cls(client=self._backend_client_map[request.backend], **request.agent_params.model_dump())


        return agent_obj
