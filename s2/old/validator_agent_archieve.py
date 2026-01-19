from common.normalization import extract_json_block
from common.prompt_loader import load_prompt
import time



class S2ValidatorAgent:

    def __init__(self, llm):
        self.llm = llm

        # Prompts are static and loaded once
        self.prompts = {
            # Single-requirement
            "atomicity": load_prompt("atomicity"),
            "clarity": load_prompt("clarity"),
            "completion_single": load_prompt("completion_single"),
            "consistency_single": load_prompt("consistency_single"),

            # # Group-level
            "completion_group": load_prompt("completion_group"),
            "consistency_group": load_prompt("consistency_group"),
            "redundancy": load_prompt("redundancy"),

            # Summary-level
            "summary": load_prompt("s2_vdp"),
        }

    # --------------------------------------------------
    # Single-requirement scope
    # --------------------------------------------------

    def validate_single(self, requirement: dict) -> dict:
        results = {}

        for key in [
            "atomicity",
            "clarity",
            "completion_single",
            "consistency_single",
        ]:
            prompt = self.prompts[key].replace(
                "{{REQUIREMENT}}", requirement["text"]
            )
            # print(f"Prompt for {key}:\n{prompt}\n")
            response = self.llm.generate(prompt)
            
            json_result = extract_json_block(response["text"])

            results[key] = {
                "output": json_result,
                "latency_ms": response["latency_ms"],
            }

        return results

    # --------------------------------------------------
    # Group scope
    # --------------------------------------------------

    def validate_group(self, group: list[dict]) -> dict:
        results = {}

        group_text = "\n".join(
            f"- {req['text']}" for req in group
        )

        for key in [
            "completion_group",
            "consistency_group",
            "redundancy",
        ]:
            prompt = self.prompts[key].replace(
                "{{REQUIREMENT}}", group_text
            )

            response = self.llm.generate(prompt)
            
            json_result = extract_json_block(response["text"])

            results[key] = {
                "output": json_result,
                "latency_ms": response["latency_ms"],
            }

        return results

    # --------------------------------------------------
    # Validation summary scope
    # --------------------------------------------------
    def summarize_validation(self, validation_results: dict, requirement) -> dict:
        prompt = self.prompts["summary"].replace(
            "{{REQUIREMENT}}", str(requirement)
        ).replace("{{VALIDATION_RESULTS}}", str(validation_results))
        response = self.llm.generate(prompt)

        json_block = extract_json_block(response["text"])

        validation_summary = {
            "output": json_block,
            "latency_ms": response["latency_ms"],
        }

        # Placeholder for future summary logic
        return validation_summary

    # --------------------------------------------------
    # Entry point
    # --------------------------------------------------

    def run(self, requirement: dict | None, group: list[dict] | None, mode: str) -> dict:
        if mode == "single":
            if requirement is None:
                raise ValueError("Single scope requested but no requirement provided")
            startTime = time.perf_counter()
            results = self.validate_single(requirement)
            summary = self.summarize_validation(results, requirement)
            endTime = time.perf_counter()
            flowlatency = int((endTime - startTime))

            return {
                "mode": "single",
                "req_id": requirement["req_id"],
                "requirement_text": requirement["text"],
                "source": requirement["source"],
                "results": results,
                "summary": summary,
                "flow_latency_seconds": flowlatency
            }

        elif mode == "group":
            if not group:
                raise ValueError("Group mode requested but no group context provided")
            startTime = time.perf_counter()
            results = self.validate_group(group)
            summary = self.summarize_validation(results, group)
            endTime = time.perf_counter()
            flowlatency = int((endTime - startTime))

            return {
                "mode": "group",
                "group_id": group[0]["group_id"],
                "source": group[0]["source"],
                "requirement_ids": [r["req_id"] for r in group],
                "requirement_texts": [r["text"] for r in group],
                "results": results,
                "summary": summary,
                "flow_latency_seconds": flowlatency
            }

        else:
            raise ValueError(f"Unknown S2 mode: {mode}")

    