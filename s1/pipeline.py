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
    
    def normalize_output(self, result:str) -> dict:
        self.logger.info(f"Normalizing output ")
        self.logger.debug(f"Raw LLM output: {result}")
        json_result = extract_json_block(result["text"])
        by_agent = {
            a["dimension"]: a["status"]
            for a in json_result["agents"]
        }
        json_result.pop("agents", None)
        json_result.pop("agent", None)
        json_result["by_agent"] = by_agent
        return json_result
    
    def run(self, requirmets:RequirementSet, mode:str):
        results = []
        if mode == "single":
            for req in requirmets.requirements:
                prompt = self.single_prompt.replace(
                "{{REQUIREMENT}}", req.text
                )
                result = self.llm.generate(prompt)
                normalized_result = self.normalize_output(result)
                results.append(normalized_result)
        elif mode == "group":
            prompt = self.group_prompt.replace(
            "{{REQUIREMENT}}", requirmets.join_requirements()
            )
            result = self.llm.generate(prompt)
            normalized_result = self.normalize_output(result)
            results.append(normalized_result)
        return results
  