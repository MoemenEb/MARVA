import logging
from utils.normalization import extract_json_block
from common.prompt_loader import load_prompt


class ValidatorAgent:
    """Handles validation operations for requirements at single and group scope."""

    def __init__(self, llm):
        self.llm = llm
        self.prompts = {
            # Single-requirement
            "atomicity": load_prompt("atomicity"),
            "clarity": load_prompt("clarity"),
            "completion_single": load_prompt("completion_single"),
            "consistency_single": load_prompt("consistency_single"),
            # Group-level
            "completion_group": load_prompt("completion_group"),
            "consistency_group": load_prompt("consistency_group"),
            "redundancy": load_prompt("redundancy"),
        }
        self.logger = logging.getLogger("marva.s2.pipeline")


    def run(self, prompt_key: str, content: str) -> dict:
        """Execute validation using specified prompt and content."""
        prompt = self.prompts[prompt_key].replace("{{REQUIREMENT}}", content)
        response = self.llm.generate(prompt)
        json_result = extract_json_block(response["text"])
        return {
            "output": json_result,
            "latency_ms": response["latency_ms"],
        }

    def validate_single(self, requirement) -> dict:
        """Validate a single requirement across multiple criteria."""
        self.logger.info(f"Validating single requirement: {requirement.id}")
        self.logger.debug(f"Requirement text: {requirement.text}")
        return {
            key: self.run(key, requirement.text)
            for key in ["atomicity", "clarity", "completion_single"]
        }

    def validate_group(self, group) -> dict:
        """Validate a group of requirements for completion, consistency, and redundancy."""
        group_text = "\n".join(f"- {req.text}" for req in group.requirements)
        
        return {
            key: self.run(key, group_text)
            for key in ["completion_group", "consistency_group", "redundancy"]
        }
    
    def execute(self, mode: str, requirement = None, group = None) -> dict:
        """Execute validation based on mode."""
        if mode == "single" and requirement is not None:
            return self.validate_single(requirement)
        elif mode == "group" and group is not None:
            return self.validate_group(group)
        else:
            raise ValueError("Invalid mode or missing requirement/group data.")
