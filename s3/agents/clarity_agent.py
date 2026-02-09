from s3.agents.base import BaseValidationAgent
from utils.normalization import extract_json_block
from entity.agent import AgentResult


class ClarityAgent(BaseValidationAgent):
    """
    Clarity validation agent with true system prompt caching.

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
        Initialize the clarity agent.

        Args:
            llm: CachedOllamaClient instance (already initialized with system prompt)
            prompts: Dict with 'task' prompt template (system prompt is in llm)
        """
        super().__init__(llm)
        self.prompts = prompts

    def run(self, input_data: dict) -> dict:
        """
        Validate a requirement for clarity.

        Sends only the task prompt (with requirement). System prompt context
        is cached in the LLM client and reused automatically.

        Args:
            input_data: Dict containing 'requirement' object with .text attribute

        Returns:
            Dict with 'clarity' key mapping to AgentResult
        """
        requirement_text = input_data["requirement"].text

        # Build task prompt with the specific requirement
        task_prompt = self.prompts["task"].replace("{{REQUIREMENT}}", requirement_text)

        # Call LLM with ONLY task prompt (system context cached)
        response = self.llm.generate(task_prompt)

        # Handle execution status
        if response["execution_status"] != "SUCCESS":
            return {
                "clarity": AgentResult(
                    agent="clarity",
                    status="FLAG",
                    issues=[]
                )
            }

        # Extract and parse response
        response_text = response["text"]
        result = extract_json_block(response_text)

        return {
            "clarity": AgentResult(
                agent="clarity",
                status=result.get("decision", "FLAG"),
                issues=result.get("issues", [])
            )
        }
