from typing import Dict, Protocol, runtime_checkable


@runtime_checkable
class LLMClientProtocol(Protocol):
    def generate(self, prompt: str) -> Dict: ...
