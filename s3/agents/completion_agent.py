from s3.agents.base import BaseValidationAgent
from utils.normalization import extract_json_block
from entity.agent import AgentResult


class CompletionAgent(BaseValidationAgent):
    """
    Completion validation agent with true system prompt caching.

    Uses CachedOllamaClient which sends system prompt ONCE during initialization
    and reuses the cached context for all subsequent validations.

    Architecture:
    - System prompt: Sent once in __init__ via CachedOllamaClient
    - Task prompt: Sent per validation with cached context
    - Stateless: No conversation history between validations
    - True caching: ~97% reduction in prompt tokens after first call

    Note: Supports both single and group mode. Each mode should have its own
    CachedOllamaClient instance with the appropriate system prompt.
    """

    def __init__(self, llm, prompts: dict[str, str]):
        """
        Initialize the completion agent.

        Args:
            llm: CachedOllamaClient instance (already initialized with system prompt)
            prompts: Dict with 'task' prompt template (system prompt is in llm)
        """
        super().__init__(llm)
        self.prompts = prompts

    def run(self, input_data: dict) -> dict:
        """
        Validate requirement(s) for completion.

        Sends only the task prompt (with requirement data). System prompt context
        is cached in the LLM client and reused automatically.

        Args:
            input_data: Dict containing 'mode' and requirement data

        Returns:
            Dict with agent key (e.g., 'completion_single') mapping to AgentResult
        """
        mode = input_data["mode"]

        # Build task prompt and determine output key based on mode
        task_prompt, output_key = self._build_prompt(input_data, mode)

        # Call LLM with ONLY task prompt (system context cached)
        response = self.llm.generate(task_prompt)

        # Handle execution status
        if response["execution_status"] != "SUCCESS":
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

        return {
            output_key: AgentResult(
                agent=output_key,
                status=result.get("decision", "FLAG"),
                issues=result.get("issues", [])
            )
        }

    # -------------------------------------------------
    # Prompt construction only (no logic)
    # -------------------------------------------------
    def _build_prompt(self, input_data: dict, mode: str) -> tuple[str, str]:
        """
        Build task prompt based on mode.

        Args:
            input_data: Dict containing requirement data
            mode: Validation mode ('single' or 'group')

        Returns:
            Tuple of (task_prompt, output_key)
        """
        if mode == "single":
            requirement_text = input_data["requirement"].text
            task_prompt = self.prompts["task"].replace(
                "{{REQUIREMENT}}", requirement_text
            )
            return task_prompt, "completion_single"

        if mode == "group":
            requirement_set = input_data["requirement_set"]
            task_prompt = self.prompts["task"].replace(
                "{{REQUIREMENT}}", requirement_set.join_requirements()
            )
            return task_prompt, "completion_group"

        raise ValueError(f"Unknown completion mode: {mode}")
