import logging
import time
from utils.normalization import extract_json_block
from common.prompt_loader import load_prompt
from entity.requirement_set import RequirementSet
from entity.agent import AgentResult


class ValidatorAgent:
    """Handles validation operations for requirements at single and group scope."""

    def __init__(self, llm):
        self.llm = llm
        self.build_prompt()
        self.logger = logging.getLogger("marva.s2.pipeline")


    def build_prompt(self):
        self.single_prompts = {
            "atomicity": load_prompt("s2/atomicity"),
            "clarity": load_prompt("s2/clarity"),
            "completion": load_prompt("s2/completion_single"),
        }
        self.group_prompts = {
           "completion": load_prompt("s2/completion_group"),
            "consistency": load_prompt("s2/consistency_group"),
            "redundancy": load_prompt("s2/redundancy"),
        }
        self.summary_prompt = load_prompt("s2/s2_vdp")


    def run(self, mode:str, requirement_set:RequirementSet):
        if mode == "single":
            total = len(requirement_set.requirements)
            for idx, requirement in enumerate(requirement_set.requirements, 1):
                req_start = time.perf_counter()
                self.logger.info("[%d/%d] Validating requirement '%s'", idx, total, requirement.id)
                for validation in self.single_prompts.keys():
                    val_start = time.perf_counter()
                    prompt = self.single_prompts[validation].replace("{{REQUIREMENT}}", requirement.text)
                    json_result = self.llm_run(prompt)
                    self.save_agent_result(validation, json_result, requirement.single_validations)
                    self.logger.debug("  Agent '%s' => %s (%.2fs)", validation, json_result.get("decision", "?"), time.perf_counter() - val_start)
                self.logger.debug("All validations done for requirement '%s'", requirement.id)
                summary_start = time.perf_counter()
                summary = self.gen_summary(requirement, requirement.single_validations)
                self.logger.debug("Summary generation took %.2fs", time.perf_counter() - summary_start)
                requirement.final_decision = summary["final_status"]
                requirement.recommendation = summary["recommendations"]
                req_elapsed = time.perf_counter() - req_start
                requirement.duration_seconds = round(req_elapsed, 3)
                self.logger.info("[%d/%d] Requirement '%s' => %s (%.2fs)", idx, total, requirement.id, requirement.final_decision, req_elapsed)

        elif mode == "group":
            self.logger.info("Running group validation for %d requirements", len(requirement_set.requirements))
            group_start = time.perf_counter()
            for validation in self.group_prompts.keys():
                val_start = time.perf_counter()
                prompt = self.group_prompts[validation].replace("{{REQUIREMENT}}", requirement_set.join_requirements())
                json_result = self.llm_run(prompt)
                self.save_agent_result(validation, json_result, requirement_set.group_validations)
                self.logger.debug("Agent '%s' => %s (%.2fs)", validation, json_result.get("decision", "?"), time.perf_counter() - val_start)
            summary_start = time.perf_counter()
            summary = self.gen_summary(requirement_set.join_requirements(), requirement_set.group_validations)
            self.logger.debug("Summary generation took %.2fs", time.perf_counter() - summary_start)
            requirement_set.final_decision = summary["final_status"]
            requirement_set.recommendations = summary["recommendations"]
            group_elapsed = time.perf_counter() - group_start
            self.logger.info("Group validation => %s (%.2fs)", requirement_set.final_decision, group_elapsed)

        else:
            raise ValueError("Invalid mode or missing requirement/group data.")

    def gen_summary(self,requirements:str, validations:dict):
        prompt = self.summary_prompt.replace(
            "{{REQUIREMENT}}", str(requirements)
        ).replace("{{VALIDATION_RESULTS}}", str(validations))
        json_block = self.llm_run(prompt)
        return json_block

    def llm_run(self,prompt:str):
        t0 = time.perf_counter()
        response = self.llm.generate(prompt)
        elapsed = time.perf_counter() - t0
        if response["execution_status"] != "SUCCESS":
            self.logger.warning("LLM call failed after %.2fs: %s - %s", elapsed, response['execution_status'], response.get('error'))
            return {"decision": "FLAG", "issues": []}
        self.logger.debug("LLM call succeeded (%.2fs, %dms reported)", elapsed, response.get("latency_ms", 0))
        return extract_json_block(response["text"])

    def save_agent_result(self, validation, json_result, validation_list):
        agent_result = AgentResult(
                        agent= validation,
                        status=json_result["decision"],
                        issues= json_result["issues"]
                    )
        validation_list.append(agent_result.to_dict())
