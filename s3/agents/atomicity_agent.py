import time
from s3.agents.base import BaseValidationAgent
from utils.normalization import extract_json_block
from entity.agent import AgentResult


class AtomicityAgent(BaseValidationAgent):
    """
    Atomicity validation agent with true system prompt caching.

    Uses CachedOllamaClient which sends system prompt ONCE during initialization
    and reuses the cached context for all subsequent validations.

    Architecture:
    - System prompt: Sent once in __init__ via CachedOllamaClient
    - Task prompt: Sent per validation with cached context
    - Stateless: No conversation history between different requirements
    - True caching: ~97% reduction in prompt tokens after first call
    """

    def __init__(self, llm, prompts: dict[str, str]):
        """
        Initialize the atomicity agent.

        Args:
            llm: CachedOllamaClient instance (already initialized with system prompt)
            prompts: Dict with 'task' prompt template (system prompt is in llm)
        """
        super().__init__(llm)
        self.prompts = prompts

    def run(self, input_data: dict) -> dict:
        """
        Validate a requirement for atomicity.

        Sends only the task prompt (with requirement). System prompt context
        is cached in the LLM client and reused automatically.

        Args:
            input_data: Dict containing 'requirement' object with .text attribute

        Returns:
            Dict with 'atomicity' key mapping to AgentResult
        """
        requirement_text = input_data["requirement"].text
        self.logger.debug("Running atomicity validation")

        # Build task prompt with the specific requirement
        task_prompt = self.prompts["task"].replace("{{REQUIREMENT}}", requirement_text)

        # Call LLM with ONLY task prompt (system context cached)
        t0 = time.perf_counter()
        response = self.llm.generate(task_prompt)
        llm_elapsed = time.perf_counter() - t0

        # Handle execution status
        if response["execution_status"] != "SUCCESS":
            self.logger.warning("Atomicity LLM call failed after %.2fs: %s", llm_elapsed, response.get("error"))
            return {
                "atomicity": AgentResult(
                    agent="atomicity",
                    status="FLAG",
                    issues=[]
                )
            }

        # Extract and parse response
        response_text = response["text"]
        result = extract_json_block(response_text)
        status = result.get("decision", "FLAG")

        self.logger.debug("Atomicity result: %s (LLM %.2fs, %dms reported)", status, llm_elapsed, response.get("latency_ms", 0))

        return {
            "atomicity": AgentResult(
                agent="atomicity",
                status=status,
                issues=result.get("issues", [])
            )
        }
