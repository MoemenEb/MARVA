from s3.agents.base import BaseValidationAgent
from utils.normalization import extract_json_block
from entity.agent import AgentResult


class DecisionAgent(BaseValidationAgent):
    """
    Decision agent with true system prompt caching.

    Uses CachedOllamaClient which sends system prompt ONCE during initialization
    and reuses the cached context for all subsequent decisions.

    Architecture:
    - System prompt: Sent once in __init__ via CachedOllamaClient
    - Task prompt: Sent per decision with cached context
    - Stateless: No conversation history between decisions
    - True caching: ~97% reduction in prompt tokens after first call
    """

    def __init__(self, llm, prompts: dict[str, str]):
        """
        Initialize the decision agent.

        Args:
            llm: CachedOllamaClient instance (already initialized with system prompt)
            prompts: Dict with 'task' prompt template (system prompt is in llm)
        """
        super().__init__(llm)
        self.prompts = prompts

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
            if v.get("agent") == "atomicity" and v.get("status") == "FAIL":
                return "FAIL"

        if any(v.get("status") == "FLAG" for v in validations):
            return "FLAG"

        return "PASS"

    # -------------------------------------------------
    # Task 3 — Recommendations (LLM-based)
    # -------------------------------------------------
    def _recommendations(self, state: dict, validations: list[dict], mode: str) -> list[str]:
        """
        Generate recommendations using LLM with cached system prompt.

        Sends only the task prompt (with mode, requirements, and issues).
        System prompt context is cached in the LLM client and reused automatically.

        Args:
            state: Current state with requirement data
            validations: List of validation results
            mode: Validation mode ('single' or 'group')

        Returns:
            List of recommendation strings
        """
        issues = self._collect_issues(validations)
        if not issues:
            return []

        requirements_text = self._format_requirements(state, mode)

        # Build task prompt with dynamic data (system prompt is cached in LLM)
        task_prompt = (
            self.prompts["task"]
            .replace("{{MODE}}", mode)
            .replace("{{REQUIREMENTS}}", requirements_text)
            .replace("{{ISSUES}}", issues)
        )

        # Call LLM with ONLY task prompt (system context cached)
        response = self.llm.generate(task_prompt)
        if response["execution_status"] != "SUCCESS":
            return []

        # Extract and parse response
        parsed = extract_json_block(response["text"])
        return parsed.get("recommendations", [])

    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------
    def _collect_issues(self, validations: list[dict]) -> str:
        lines = []
        for v in validations:
            if v.get("issues"):
                lines.append(f"{v['agent']}: {v['issues']}")
        return "\n".join(lines)

    def _format_requirements(self, state: dict, mode: str) -> str:
        if mode == "single":
            return state["requirement"].text

        return state["requirement_set"].join_requirements()
