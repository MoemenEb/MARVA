import time
from s3.agents.base import BaseValidationAgent
from utils.normalization import extract_json_block
from entity.agent import AgentResult


class ConsistencyAgent(BaseValidationAgent):
    """
    Consistency validation agent with true system prompt caching.

    Uses CachedOllamaClient which sends system prompt ONCE during initialization
    and reuses the cached context for all subsequent validations.

    Architecture:
    - System prompt: Sent once in __init__ via CachedOllamaClient
    - Task prompt: Sent per validation with cached context
    - Stateless: No conversation history between different requirement sets
    - True caching: ~97% reduction in prompt tokens after first call

    Note: Currently only supports group mode validation.
    """

    def __init__(self, llm, prompts: dict[str, str]):
        """
        Initialize the consistency agent.

        Args:
            llm: CachedOllamaClient instance (already initialized with system prompt)
            prompts: Dict with 'task' prompt template (system prompt is in llm)
        """
        super().__init__(llm)
        self.prompts = prompts

    def run(self, input_data: dict) -> dict:
        """
        Validate a requirement set for consistency.

        Sends only the task prompt (with requirement set). System prompt context
        is cached in the LLM client and reused automatically.

        Args:
            input_data: Dict containing 'mode' and 'requirement_set'

        Returns:
            Dict with agent key (e.g., 'consistency_group') mapping to AgentResult
        """
        mode = input_data["mode"]

        # Build task prompt and determine output key based on mode
        task_prompt, output_key = self._build_prompt(input_data, mode)
        self.logger.debug("Running consistency validation (mode=%s)", mode)

        # Call LLM with ONLY task prompt (system context cached)
        t0 = time.perf_counter()
        response = self.llm.generate(task_prompt)
        llm_elapsed = time.perf_counter() - t0

        # Handle execution status
        if response["execution_status"] != "SUCCESS":
            self.logger.warning("Consistency LLM call failed after %.2fs: %s", llm_elapsed, response.get("error"))
            return {
                output_key: AgentResult(
                    agent=output_key,
                    status="FLAG",
                    issues=[]
                )
            }

        # Extract and parse response
        response_text = response["text"]
        result = extract_json_block(response_text)
        status = result.get("decision", "FLAG")

        self.logger.debug("Consistency result: %s (LLM %.2fs, %dms reported)", status, llm_elapsed, response.get("latency_ms", 0))

        return {
            output_key: AgentResult(
                agent=output_key,
                status=status,
                issues=result.get("issues", [])
            )
        }

    # -------------------------------------------------
    # Prompt construction only
    # -------------------------------------------------
    def _build_prompt(self, input_data: dict, mode: str) -> tuple[str, str]:
        """
        Build task prompt based on mode.

        Args:
            input_data: Dict containing requirement data
            mode: Validation mode ('group')

        Returns:
            Tuple of (task_prompt, output_key)
        """
        if mode == "group":
            requirement_set = input_data["requirement_set"]
            task_prompt = self.prompts["task"].replace(
                "{{REQUIREMENT}}", requirement_set.join_requirements()
            )
            return task_prompt, "consistency_group"

        raise ValueError(f"Unknown consistency mode: {mode}")
