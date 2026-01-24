from common.llm_client import LLMClient
from common.prompt_loader import load_prompt
import logging

from entity.requirement_set import RequirementSet
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


    def run_single(self, requirement: dict) -> dict:
        self.logger.info(f"Running S1 single for requirement ID: {requirement.id}")
        prompt = self.single_prompt.replace(
            "{{REQUIREMENT}}", requirement.text
        )

        response = self.llm.generate(prompt)

        return {
            "mode": "single",
            "req_id": requirement.id,
            "llm_output": response["text"],
            "latency_ms": response["latency_ms"],
        }

    def run_group(self, requirements: list[dict]) -> dict:
        self.logger.info(f"Running S1 group for requirements")
        joined_reqs = "\n".join(
            f"[{r.id}] {r.text}" for r in requirements
        )

        prompt = self.group_prompt.replace(
            "{{REQUIREMENT}}", joined_reqs
        )

        response = self.llm.generate(prompt)

        return {
            "mode": "group",
            "requirements": joined_reqs,
            "llm_output": response["text"],
            "latency_ms": response["latency_ms"],
        }
    
    def normalize_output(self, result:str) -> dict:
        self.logger.info(f"Normalizing output ")
        json_result = extract_json_block(result["llm_output"])
        by_agent = {
            a["dimension"]: a["status"]
            for a in json_result["agents"]
        }
        json_result.pop("agents", None)
        json_result.pop("agent", None)
        json_result["by_agent"] = by_agent
        return json_result

    def execute(self, requirement, mode: str) -> dict:
        self.logger.info(f"Executing S1 pipeline in {mode} mode")
        if mode == "single":
            result = self.run_single(requirement)
        elif mode == "group":
            result = self.run_group(requirement)
        else:
            raise ValueError(f"Invalid mode or missing input for mode: {mode}")
        return self.normalize_output(result)
