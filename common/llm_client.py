class LLMClient:
    def generate(self, prompt: str) -> dict:
        """
        Returns:
        {
          "text": str,
          "raw": dict,
          "model": str,
          "latency_ms": int
        }
        """
