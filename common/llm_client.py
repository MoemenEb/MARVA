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

    llm_response = {
        
                    "execution_status": "",
                    "attempts": 0,
                    "text": "",
                    "latency_ms": 0,
                    "error": None     
                }

    def generate(self, prompt: str) -> Dict:
        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "stream": False,
            "reset_session": True
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
                self.llm_response["execution_status"] = "SUCCESS"
                self.llm_response["attempts"] = attempts
                self.llm_response["text"] = data.get("response", "").strip()
                self.llm_response["latency_ms"] = elapsed
                return self.llm_response

            except requests.exceptions.Timeout:
                if attempts > self.max_retries:
                    break
                time.sleep(self.retry_backoff)

            except requests.exceptions.RequestException as e:
                elapsed = int((time.time() - start) * 1000)
                self.llm_response["execution_status"] = "ERROR"
                self.llm_response["attempts"] = attempts
                self.llm_response["error"] = str(e)
                self.llm_response["text"] = None
                self.llm_response["latency_ms"] = elapsed
                return self.llm_response

        elapsed = int((time.time() - start) * 1000)
        self.llm_response["execution_status"] = "TIMEOUT"
        self.llm_response["attempts"] = attempts
        self.llm_response["text"] = None
        self.llm_response["latency_ms"] = elapsed
        self.llm_response["error"] = "Max retries exceeded due to timeout."
        return self.llm_response