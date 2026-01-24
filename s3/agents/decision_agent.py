from s3.agents.base import BaseValidationAgent
from utils.normalization import extract_json_block


class DecisionAgent(BaseValidationAgent):

    def __init__(self, llm, prompt: str):
        super().__init__(llm)
        self.prompt = prompt

    # -------------------------------------------------
    # Entry point
    # -------------------------------------------------
    def run(self, state: dict) -> dict:
        mode = state["mode"]

        aggregated = self._collect_results(state, mode)
        final_decision = self._final_decision(aggregated)
        recommendations = self._recommendations(state, aggregated, mode)

        return {
            "decision": {
                "agent": "validation decision agent",
                # "mode": mode,
                "final_decision": final_decision,
                "by_agent": aggregated,
                "recommendations": recommendations,
            }
        }

    # -------------------------------------------------
    # Task 1 — Collect & aggregate
    # -------------------------------------------------
    def _collect_results(self, state: dict, mode: str) -> dict:
        results = {}
        if mode == "single":
            if "atomicity" in state:
                results["atomicity"] = state["atomicity"]["decision"]

            if state.get("atomicity", {}).get("decision") != "FAIL":
                for key in ["clarity", "completion_single", "consistency_single"]:
                    if key in state:
                        results[key] = state[key]["decision"]

        elif mode == "group":
            for key in ["redundancy", "completion_group", "consistency_group"]:
                if key in state:
                    results[key] = state[key]["decision"]

        else:
            raise ValueError(f"Unknown mode: {mode}")

        return results

    # -------------------------------------------------
    # Task 2 — Final decision
    # -------------------------------------------------
    def _final_decision(self, aggregated: dict) -> str:
        if aggregated.get("atomicity") == "FAIL":
                return "FAIL"

        if any(decision == "FLAG" for decision in aggregated.values()):
            return "FLAG"

        return "PASS"

    # -------------------------------------------------
    # Task 3 — Recommendations (LLM-based)
    # -------------------------------------------------
    def _recommendations(self, state: dict, aggregated: dict, mode: str) -> list[str]:
        issues = self._collect_issues(state)
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
    def _collect_issues(self, state: dict) -> str:
        lines = []
        for key, value in state.items():
            if isinstance(value, dict) and value.get("issues"):
                lines.append(f"{key}: {value['issues']}")

        return "\n".join(lines)

    def _format_requirements(self, state: dict, mode: str) -> str:
        if mode == "single":
            return state["requirement"].text

        return "\n".join(
            f"- {req.text}" for req in state["group"]
        )
