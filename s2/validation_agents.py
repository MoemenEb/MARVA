import logging
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
            "atomicity": load_prompt("atomicity"),
            "clarity": load_prompt("clarity"),
            "completion": load_prompt("completion_single"),
        }
        self.group_prompts = {
           "completion": load_prompt("completion_group"),
            "consistency": load_prompt("consistency_group"),
            "redundancy": load_prompt("redundancy"), 
        }
        self.summary_prompt = load_prompt("s2_vdp")


    def run(self, mode:str, requirement_set:RequirementSet):
        if mode == "single":
            for requirement in requirement_set.requirements:
                for validation in self.single_prompts.keys():
                    prompt = self.single_prompts[validation].replace("{{REQUIREMENT}}", requirement.text)
                    json_result = self.llm_run(prompt)
                    self.save_agent_result(validation, json_result, requirement.single_validations)
                self.logger.debug(f"Validation done for requirement: {requirement.to_dict()}" )
                summary = self.gen_summary(requirement,requirement.single_validations)
                self.logger.debug(f"Summary : {summary}")
                requirement.final_decision = summary["final_status"]
                requirement.recommendation = summary["recommendations"]
                self.logger.debug(f"final requirement : {requirement.to_dict()}")
                
        elif mode == "group":
            for validation in self.group_prompts.keys():
                prompt = self.group_prompts[validation].replace("{{REQUIREMENT}}", requirement_set.join_requirements())
                json_result = self.llm_run(prompt)
                self.save_agent_result(validation, json_result, requirement_set.group_validations)
            summary = self.gen_summary(requirement_set.join_requirements(),requirement_set.group_validations)
            requirement_set.final_decision = summary["final_status"]
            requirement_set.recommendations = summary["recommendations"]
            self.logger.debug(f"Final requirement : {requirement_set.to_dict()}")

        else:
            raise ValueError("Invalid mode or missing requirement/group data.")
        
    def gen_summary(self,requirements:str, validations:dict):
        prompt = self.summary_prompt.replace(
            "{{REQUIREMENT}}", str(requirements)
        ).replace("{{VALIDATION_RESULTS}}", str(validations))
        json_block = self.llm_run(prompt)
        return json_block        

    def llm_run(self,prompt:str):
        response = self.llm.generate(prompt)
        if response["execution_status"] != "SUCCESS":
            self.logger.warning(f"LLM call failed: {response['execution_status']} - {response.get('error')}")
            return {"decision": "FLAG", "issues": []}
        return extract_json_block(response["text"])
    
    def save_agent_result(self, validation, json_result, validation_list):
        agent_result = AgentResult(
                        agent= validation,
                        status=json_result["decision"],
                        issues= json_result["issues"]
                    )
        validation_list.append(agent_result.to_dict())

