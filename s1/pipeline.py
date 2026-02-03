from common.llm_client import LLMClient
from common.prompt_loader import load_prompt
import logging

from entity.requirement_set import RequirementSet
from entity.agent import AgentResult
from entity.agent_set import AgentSet
from utils.normalization import extract_json_block


class S1Pipeline:
    SINGLE_PROMPT_PATH = "s1/s1_single"
    GROUP_PROMPT_PATH = "s1/s1_group"
    LOGGER = "marva.s1.pipeline"

    def __init__(
        self,
        llm: LLMClient
    ):
        self.llm = llm
        self.single_prompt = load_prompt(self.SINGLE_PROMPT_PATH)
        self.group_prompt = load_prompt(self.GROUP_PROMPT_PATH)
        self.logger = logging.getLogger(self.LOGGER)
        self.agents = []

    def normalize_output(self, result:str):
        self.logger.info(f"Normalizing output ")
        json_result = extract_json_block(result["text"])
        self.save_agent_result(json_result["agents"])
        return json_result["status"], json_result["recommendations"]
    
    def run(self, requirement_set:RequirementSet, mode:str):
        if mode == "single":
            for requirement in requirement_set.requirements:
                # prep prompt
                prompt = self.single_prompt.replace("{{REQUIREMENT}}", requirement.text)
                normalized_result = self.prompt_run(prompt)
                # Save result.
                requirement.final_decision,requirement.recommendation = normalized_result
                age = AgentSet(self.agents)
                requirement.single_validations = age.agents_list()
        elif mode == "group":
            # prep prompt
            prompt = self.group_prompt.replace("{{REQUIREMENT}}", requirement_set.join_requirements())
            normalized_result = self.prompt_run(prompt)
            # Save result
            requirement_set.final_decision, requirement_set.recommendations = normalized_result
            age = AgentSet(self.agents)
            requirement_set.group_validations = age.agents_list()
    
  
    def prompt_run(self, prompt):
        return self.normalize_output(
            self.llm.generate(prompt)
            )
    
    def save_agent_result(self, agents_json):
        self.agents.clear()
        for agent in agents_json:
            self.agents.append( AgentResult(
                agent= agent["dimension"],
                status= agent["status"],
                issues= agent["issues"]
            ))