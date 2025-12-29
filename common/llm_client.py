import requests
import time

class LLMClient:
    def __init__(self, host: str, model: str, timeout: int = 60):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: str) -> dict:
        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        start = time.time()
        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        elapsed = int((time.time() - start) * 1000)

        data = response.json()

        return {
            "text": data.get("response", "").strip(),
            "raw": data,
            "model": self.model,
            "latency_ms": elapsed
        }
