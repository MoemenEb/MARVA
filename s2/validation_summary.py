from common.normalization import extract_json_block
from common.prompt_loader import load_prompt

class ValidatorSummary:
    """Handles summarization of validation results."""

    def __init__(self, llm):
        self.llm = llm
        self.summary_prompt = load_prompt("s2_vdp")

    def summarize(self, validation_results: dict, requirement) -> dict:
        """Generate a summary of validation results."""
        prompt = self.summary_prompt.replace(
            "{{REQUIREMENT}}", str(requirement)
        ).replace("{{VALIDATION_RESULTS}}", str(validation_results))
        
        response = self.llm.generate(prompt)
        json_block = extract_json_block(response["text"])

        return {
            "output": json_block,
            "latency_ms": response["latency_ms"],
        }
