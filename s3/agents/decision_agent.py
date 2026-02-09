import time
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
        overall_start = time.perf_counter()
        self.logger.debug("Decision agent started (mode=%s)", mode)

        validations = self._collect_validations(state, mode)
        self.logger.debug("Collected %d validations: %s", len(validations), [v.get("agent") for v in validations])

        final_decision = self._final_decision(validations)
        self.logger.info("Final decision: %s", final_decision)

        if final_decision == "PASS":
            self.logger.debug("Final decision is PASS — skipping recommendation generation")
            recommendations = []
        else:
            t0 = time.perf_counter()
            recommendations = self._recommendations(state, validations, mode)
            rec_elapsed = time.perf_counter() - t0
            self.logger.debug("Recommendations generated in %.2fs (%d items)", rec_elapsed, len(recommendations))

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

        overall_elapsed = time.perf_counter() - overall_start
        self.logger.debug("Decision agent completed in %.2fs", overall_elapsed)

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
            else:
                self.logger.warning("Missing validation result for '%s'", key)

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
            self.logger.debug("No issues found — skipping recommendation generation")
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
        t0 = time.perf_counter()
        response = self.llm.generate(task_prompt)
        llm_elapsed = time.perf_counter() - t0

        if response["execution_status"] != "SUCCESS":
            self.logger.warning("Recommendation LLM call failed after %.2fs: %s", llm_elapsed, response.get("error"))
            return []

        # Extract and parse response
        parsed = extract_json_block(response["text"])
        recs = parsed.get("recommendations", [])
        self.logger.debug("Generated %d recommendations (LLM %.2fs)", len(recs), llm_elapsed)
        return recs

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
