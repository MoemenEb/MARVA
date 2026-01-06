from common.prompt_loader import load_prompt


class S2ValidatorAgent:
    """
    S2 Validator Agent (Single Agent, Multi-Call, Single Scope)

    This agent:
    - Owns the validation workflow
    - Executes multiple SLM calls procedurally
    - Operates under exactly ONE scope per run

    All reasoning is delegated to the SLM.
    """

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
        """
        Executes ONLY single-requirement validations.

        Each validation:
        - Uses one focused prompt
        - Triggers one SLM call
        - Result is stored verbatim
        """

        results = {"scope": "single"}

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

            results[key] = {
                "output": response["text"],
                "latency_ms": response["latency_ms"],
            }

        return results

    # --------------------------------------------------
    # Group scope
    # --------------------------------------------------

    def validate_group(self, group: list[dict]) -> dict:
        """
        Executes ONLY group-level validations.

        The agent reasons over the group as a whole.
        It does not produce per-requirement judgments here.
        """

        results = {"scope": "group"}

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

            results[key] = {
                "output": response["text"],
                "latency_ms": response["latency_ms"],
            }

        return results

    # --------------------------------------------------
    # Validation summary scope
    # --------------------------------------------------
    def summarize_validation(self, validation_results: dict) -> dict:
        """
        Summarizes validation results across scopes.

        This method can be expanded to provide
        aggregated insights if needed.
        """
        prompt = self.prompts["summary"].replace(
            "{{VALIDATION_RESULTS}}", str(validation_results)
        )
        response = self.llm.generate(prompt)

        validation_summary = {
            "output": response["text"],
            "latency_ms": response["latency_ms"],
        }

        # Placeholder for future summary logic
        return validation_summary

    # --------------------------------------------------
    # Entry point
    # --------------------------------------------------

    def run(self, requirement: dict | None, group: list[dict] | None, scope: str) -> dict:
        """
        Executes S2 validation under a single scope.

        scope:
        - 'single' → per-requirement execution
        - 'group'  → per-group execution
        """

        if scope == "single":
            if requirement is None:
                raise ValueError("Single scope requested but no requirement provided")

            results = self.validate_single(requirement)
            summary = self.summarize_validation(results)

            return {
                "scope": "single",
                "req_id": requirement["req_id"],
                "source": requirement["source"],
                "group_id": requirement["group_id"],
                "results": results,
                "summary": summary
            }

        elif scope == "group":
            if not group:
                raise ValueError("Group scope requested but no group context provided")

            results = self.validate_group(group)
            summary = self.summarize_validation(results)

            return {
                "scope": "group",
                "group_id": group[0]["group_id"],
                "source": group[0]["source"],
                "requirement_ids": [r["req_id"] for r in group],
                "results": results,
                "summary": summary
            }

        else:
            raise ValueError(f"Unknown S2 scope: {scope}")
