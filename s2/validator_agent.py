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

            # Group-level
            "completion_group": load_prompt("completion_group"),
            "consistency_group": load_prompt("consistency_group"),
            "redundancy": load_prompt("redundancy"),
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
            prompt = self.prompts[key].format(
                requirement=requirement["text"]
            )

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
            prompt = self.prompts[key].format(
                requirements=group_text
            )

            response = self.llm.generate(prompt)

            results[key] = {
                "output": response["text"],
                "latency_ms": response["latency_ms"],
            }

        return results

    # --------------------------------------------------
    # Entry point
    # --------------------------------------------------

    def run(self, requirement: dict, group: list[dict] | None, scope: str) -> dict:
        """
        Executes S2 validation under a single scope.

        Parameters:
        - scope = 'single' OR 'group'
        """

        if scope == "single":
            results = self.validate_single(requirement)

        elif scope == "group":
            if group is None:
                raise ValueError("Group scope requested but no group context provided")
            results = self.validate_group(group)

        else:
            raise ValueError(f"Unknown S2 scope: {scope}")

        return {
            "req_id": requirement["req_id"],
            "source": requirement["source"],
            "group_id": requirement["group_id"],
            "results": results
        }
