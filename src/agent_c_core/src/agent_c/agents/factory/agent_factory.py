import logging

from agent_c.agents.factory.agent_interface import AgentInterface
from agent_c.agents.gpt import GPTChatAgent, AzureOpenAIChatAgent
from agent_c.models.agent_factory_request import AgentFactoryRequest
from agent_c.agents.claude import ClaudeChatAgent, ClaudeBedrockChatAgent

class AgentFactory:
    __backend_to_agent_map = {
        "azure_openai": AzureOpenAIChatAgent,
        "openai": GPTChatAgent,
        "claude": ClaudeChatAgent,
        "claude_aws": ClaudeBedrockChatAgent
    }

    def __init__(self, **kwargs):
        backends = kwargs.get('backends', ['openai', 'claude', 'azure_openai', 'claude_aws'])
        self._backend_client_map = {}

        for backend in backends:
            try:
                backend_cls = self.__backend_to_agent_map[backend]
                self._backend_client_map[backend] = kwargs.get(f"{backend}_client", backend_cls.default_client())
            except Exception as e:
                logging.warning(f"Failed to initialize backend {backend} with error {e}")
                continue

        if len(self._backend_client_map) == 0:
            raise ValueError("No valid backends available")


    def __backend_for_request(self, request: AgentFactoryRequest):
        if request.agent_params.backend not in self.__backend_to_agent_map:
            raise ValueError(f"Invalid backend  {request.agent_params.backend}")
        elif request.agent_params.backend not in self._backend_client_map:
            raise ValueError(f"No client available for backend {request.agent_params.backend}")

        return self.__backend_to_agent_map.get(request.agent_params.backend)


    def create_agent(self, request: AgentFactoryRequest):
        agent_cls = self.__backend_for_request(request)
        agent_client = self._backend_client_map[request.agent_params.backend]
        agent_obj = agent_cls(client=agent_client, **request.agent_params.model_dump())

        return  AgentInterface(agent_obj)
