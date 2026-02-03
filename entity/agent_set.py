from .agent import AgentResult

class AgentSet:
    def __init__(self, agents: list[AgentResult]):
        self.agent_list = agents

    def agents_list(self):
        agents = []
        for agent in self.agent_list:
            agents.append(agent.to_dict())
        return agents