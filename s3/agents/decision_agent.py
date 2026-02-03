from s3.agents.base import BaseValidationAgent
from utils.normalization import extract_json_block
from entity.agent import AgentResult


class DecisionAgent(BaseValidationAgent):

    def __init__(self, llm, prompt: str):
        super().__init__(llm)
        self.prompt = prompt

    # -------------------------------------------------
    # Entry point
    # -------------------------------------------------
    def run(self, state: dict) -> dict:
        mode = state["mode"]

        validations = self._collect_validations(state, mode)
        final_decision = self._final_decision(validations)
        recommendations = self._recommendations(state, validations, mode)

        # Update the entity directly
        if mode == "single":
            req = state["requirement"]
            req.single_validations = validations
            req.final_decision = final_decision
            req.recommendation = recommendations
        else:
            req_set = state["requirement_set"]
            req_set.group_validations = validations
            req_set.final_decision = final_decision
            req_set.recommendations = recommendations

        return {
            "decision": AgentResult(
                agent="Decision Agent",
                status=final_decision
            )
        }

    # -------------------------------------------------
    # Task 1 — Collect validations
    # -------------------------------------------------
    def _collect_validations(self, state: dict, mode: str) -> list[dict]:
        validations = []

        if mode == "single":
            keys = ["atomicity", "clarity", "completion_single"]
        elif mode == "group":
            keys = ["redundancy", "completion_group", "consistency_group"]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        for key in keys:
            if key in state and isinstance(state[key], AgentResult):
                validations.append(state[key].to_dict())

        return validations

    # -------------------------------------------------
    # Task 2 — Final decision
    # -------------------------------------------------
    def _final_decision(self, validations: list[dict]) -> str:
        for v in validations:
            if v.get("Agent") == "atomicity" and v.get("Status") == "FAIL":
                return "FAIL"

        if any(v.get("Status") == "FLAG" for v in validations):
            return "FLAG"

        return "PASS"

    # -------------------------------------------------
    # Task 3 — Recommendations (LLM-based)
    # -------------------------------------------------
    def _recommendations(self, state: dict, validations: list[dict], mode: str) -> list[str]:
        issues = self._collect_issues(validations)
        if not issues:
            return []

        requirements_text = self._format_requirements(state, mode)

        prompt = (
            self.prompt
            .replace("{{MODE}}", mode)
            .replace("{{REQUIREMENTS}}", requirements_text)
            .replace("{{ISSUES}}", issues)
        )

        raw = self.llm.generate(prompt)["text"]
        parsed = extract_json_block(raw)

        return parsed.get("recommendations", [])

    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------
    def _collect_issues(self, validations: list[dict]) -> str:
        lines = []
        for v in validations:
            if v.get("Issues"):
                lines.append(f"{v['Agent']}: {v['Issues']}")
        return "\n".join(lines)

    def _format_requirements(self, state: dict, mode: str) -> str:
        if mode == "single":
            return state["requirement"].text

        return state["requirement_set"].join_requirements()
