from common.llm_client import LLMClient
from common.prompt_loader import load_prompt
import logging
import time

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
        self.logger.debug("Normalizing LLM output")
        if result["execution_status"] != "SUCCESS":
            self.logger.warning("LLM call failed: %s - %s", result['execution_status'], result.get('error'))
            return "FLAG", []
        json_result = extract_json_block(result["text"])
        self.save_agent_result(json_result["agents"])
        return json_result["status"], json_result["recommendations"]

    def run(self, requirement_set:RequirementSet, mode:str):
        if mode == "single":
            total = len(requirement_set.requirements)
            for idx, requirement in enumerate(requirement_set.requirements, 1):
                req_start = time.perf_counter()
                self.logger.info("[%d/%d] Validating requirement '%s'", idx, total, requirement.id)
                # prep prompt
                prompt = self.single_prompt.replace("{{REQUIREMENT}}", requirement.text)
                normalized_result = self.prompt_run(prompt)
                # Save result.
                requirement.final_decision,requirement.recommendation = normalized_result
                age = AgentSet(self.agents)
                requirement.single_validations = age.agents_list()
                req_elapsed = time.perf_counter() - req_start
                self.logger.info("[%d/%d] Requirement '%s' => %s (%.2fs)", idx, total, requirement.id, requirement.final_decision, req_elapsed)
        elif mode == "group":
            self.logger.info("Running group validation for %d requirements", len(requirement_set.requirements))
            group_start = time.perf_counter()
            # prep prompt
            prompt = self.group_prompt.replace("{{REQUIREMENT}}", requirement_set.join_requirements())
            normalized_result = self.prompt_run(prompt)
            # Save result
            requirement_set.final_decision, requirement_set.recommendations = normalized_result
            age = AgentSet(self.agents)
            requirement_set.group_validations = age.agents_list()
            group_elapsed = time.perf_counter() - group_start
            self.logger.info("Group validation => %s (%.2fs)", requirement_set.final_decision, group_elapsed)


    def prompt_run(self, prompt):
        t0 = time.perf_counter()
        result = self.llm.generate(prompt)
        self.logger.debug("LLM call took %dms (status=%s)", result.get("latency_ms", 0), result.get("execution_status"))
        return self.normalize_output(result)

    def save_agent_result(self, agents_json):
        self.agents.clear()
        for agent in agents_json:
            self.agents.append( AgentResult(
                agent= agent["dimension"],
                status= agent["status"],
                issues= agent["issues"]
            ))
        self.logger.debug("Saved %d agent results: %s", len(self.agents), [a.agent for a in self.agents])
