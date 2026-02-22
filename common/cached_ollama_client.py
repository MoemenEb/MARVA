"""
Cached Ollama Client with System Prompt Context Caching

This client sends the system prompt ONCE during initialization and caches
the resulting context. Subsequent calls only send user messages + cached context,
achieving true token-level caching for maximum efficiency.

Compatible with the agent architecture and can be used as a drop-in replacement
for ChatOllama or LLMClient in agent initialization.
"""

import re
import requests
import time
import logging
from typing import Dict, List, Optional

logger = logging.getLogger("marva.cached_ollama")


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

    _THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

    def __init__(
        self,
        model: str,
        base_url: str,
        system_prompt: str,
        temperature: float = 0.0,
        num_predict: int = 1024,
        timeout: int = 60,
        max_retries: int = 3,
        disable_think: bool = True,
    ):
        """
        Initialize client and cache system prompt context.

        Args:
            model: Ollama model name (e.g., "qwen3:1.7b")
            base_url: Ollama API base URL (e.g., "http://localhost:11434")
            system_prompt: System prompt to cache (sent once)
            temperature: LLM temperature setting
            num_predict: Maximum tokens to generate per call
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts on failure
            disable_think: If True, passes think=False to Ollama (suppresses qwen3
                           chain-of-thought blocks that would otherwise consume the
                           token budget and truncate the JSON response). Safe to set
                           on models that don't support this option — Ollama ignores it.
        """
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.num_predict = num_predict
        self.timeout = timeout
        self.max_retries = max_retries
        self.system_prompt = system_prompt
        self.disable_think = disable_think

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
                "temperature": self.temperature,
                "num_predict": 1,  # minimize wasted generation, we only need the context
                **({"think": False} if self.disable_think else {}),
            }
        }

        logger.debug("Caching system prompt (model=%s, prompt_len=%d)", self.model, len(self.system_prompt))
        start = time.perf_counter()

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()

            # Extract and return context
            context = result.get("context", [])
            elapsed_ms = (time.perf_counter() - start) * 1000

            if not context:
                logger.warning("No context returned from system prompt init (model=%s, %.0fms)", self.model, elapsed_ms)
            else:
                logger.info("System prompt cached (model=%s, context_tokens=%d, %.0fms)", self.model, len(context), elapsed_ms)

            return context

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error("Failed to cache system prompt (model=%s, %.0fms): %s", self.model, elapsed_ms, e)
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
                "temperature": self.temperature,
                "num_predict": self.num_predict,
                **({"think": False} if self.disable_think else {}),
            }
        }

        logger.debug("Cached generate called (prompt_len=%d, cached_context=%d tokens)", len(prompt), len(self.system_context))

        for attempt in range(1, self.max_retries + 1):
            try:
                start_time = time.time()

                response = requests.post(url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                result = response.json()

                latency_ms = int((time.time() - start_time) * 1000)
                text = result.get("response", "")
                logger.debug("Raw Ollama response (len=%d): %r", len(text), text[:300])
                # Strip thinking blocks (e.g. qwen3 <think>...</think>)
                text = self._THINK_RE.sub("", text).strip()
                logger.debug("Cached generate response (latency=%dms, response_len=%d, attempt=%d)", latency_ms, len(text), attempt)

                return {
                    "execution_status": "SUCCESS",
                    "text": text,
                    "latency_ms": latency_ms,
                    "error": None,
                    "attempts": attempt
                }

            except requests.Timeout:
                logger.warning("Cached generate timed out (attempt %d/%d, timeout=%ds)", attempt, self.max_retries, self.timeout)
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
                logger.error("Cached generate failed (attempt %d/%d): %s", attempt, self.max_retries, e)
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
