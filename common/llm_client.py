import requests
import time
from typing import Dict, Optional


class LLMClient:
    def __init__(
        self,
        host: str,
        model: str,
        timeout: int = 60,
        temperature: float = 0.0,
        max_retries: int = 1,
        retry_backoff: float = 1.5
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

    def generate(self, prompt: str, reset_session: bool = False) -> Dict:
        """
        Generate text using the LLM.

        Args:
            prompt: The input prompt text
            reset_session: If True, reset conversation context. Default False for session reuse.

        Returns:
            Dict with execution_status, attempts, text, latency_ms, and error
        """
        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "stream": False,
            "reset_session": reset_session
        }

        attempts = 0
        start = time.time()

        while attempts <= self.max_retries:
            try:
                attempts += 1

                response = requests.post(
                    url,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()

                elapsed = int((time.time() - start) * 1000)
                data = response.json()
                return {
                    "execution_status": "SUCCESS",
                    "attempts": attempts,
                    "text": data.get("response", "").strip(),
                    "latency_ms": elapsed,
                    "error": None
                }

            except requests.exceptions.Timeout:
                if attempts > self.max_retries:
                    break
                time.sleep(self.retry_backoff)

            except requests.exceptions.RequestException as e:
                elapsed = int((time.time() - start) * 1000)
                return {
                    "execution_status": "ERROR",
                    "attempts": attempts,
                    "text": None,
                    "latency_ms": elapsed,
                    "error": str(e)
                }

        elapsed = int((time.time() - start) * 1000)
        return {
            "execution_status": "TIMEOUT",
            "attempts": attempts,
            "text": None,
            "latency_ms": elapsed,
            "error": "Max retries exceeded due to timeout."
        }