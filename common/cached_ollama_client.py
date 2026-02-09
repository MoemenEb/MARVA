"""
Cached Ollama Client with System Prompt Context Caching

This client sends the system prompt ONCE during initialization and caches
the resulting context. Subsequent calls only send user messages + cached context,
achieving true token-level caching for maximum efficiency.

Compatible with the agent architecture and can be used as a drop-in replacement
for ChatOllama or LLMClient in agent initialization.
"""

import requests
import time
from typing import Dict, List, Optional


class CachedOllamaClient:
    """
    Ollama client with system prompt context caching.

    Architecture:
    - __init__: Sends system prompt once, stores returned context
    - generate: Sends only user prompt + cached context
    - Stateless: Context is immutable after initialization
    - Reusable: Can be used for any agent that needs system prompt caching

    Example:
        client = CachedOllamaClient(
            model="qwen3:1.7b",
            base_url="http://localhost:11434",
            system_prompt="You are an expert validator...",
            temperature=0.0
        )

        response = client.generate("Validate this requirement: ...")
        # System prompt tokens are cached, only user prompt is processed
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        system_prompt: str,
        temperature: float = 0.0,
        timeout: int = 60,
        max_retries: int = 3
    ):
        """
        Initialize client and cache system prompt context.

        Args:
            model: Ollama model name (e.g., "qwen3:1.7b")
            base_url: Ollama API base URL (e.g., "http://localhost:11434")
            system_prompt: System prompt to cache (sent once)
            temperature: LLM temperature setting
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts on failure
        """
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.system_prompt = system_prompt

        # Initialize system context (send system prompt once)
        self.system_context = self._initialize_system_context()

    def _initialize_system_context(self) -> List[int]:
        """
        Send system prompt to Ollama and cache the context.

        This is called once during __init__ to establish the cached context.
        The returned context contains encoded system prompt tokens that will
        be reused for all subsequent generate() calls.

        Returns:
            List of token IDs representing the cached system prompt
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": self.system_prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature
            }
        }

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()

            # Extract and return context
            context = result.get("context", [])

            if not context:
                print(f"Warning: No context returned from system prompt initialization")

            return context

        except Exception as e:
            print(f"Error initializing system context: {e}")
            return []

    def generate(self, prompt: str) -> Dict:
        """
        Generate response using cached system context.

        Only sends the user prompt + cached context. System prompt tokens
        are NOT re-sent, achieving true token-level caching.

        Args:
            prompt: User prompt to send

        Returns:
            Dict with 'execution_status', 'text', 'latency_ms', and 'error' keys
            (compatible with LLMClient format for backward compatibility)
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "context": self.system_context,  # Reuse cached context
            "stream": False,
            "options": {
                "temperature": self.temperature
            }
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                start_time = time.time()

                response = requests.post(url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                result = response.json()

                latency_ms = int((time.time() - start_time) * 1000)

                return {
                    "execution_status": "SUCCESS",
                    "text": result.get("response", ""),
                    "latency_ms": latency_ms,
                    "error": None,
                    "attempts": attempt
                }

            except requests.Timeout:
                if attempt == self.max_retries:
                    return {
                        "execution_status": "TIMEOUT",
                        "text": "",
                        "latency_ms": self.timeout * 1000,
                        "error": "Request timeout",
                        "attempts": attempt
                    }
                continue

            except Exception as e:
                if attempt == self.max_retries:
                    return {
                        "execution_status": "ERROR",
                        "text": "",
                        "latency_ms": 0,
                        "error": str(e),
                        "attempts": attempt
                    }
                continue

        return {
            "execution_status": "ERROR",
            "text": "",
            "latency_ms": 0,
            "error": "Max retries exceeded",
            "attempts": self.max_retries
        }
