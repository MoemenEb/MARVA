from abc import ABC, abstractmethod
from typing import Any
import logging

from common.llm_client import LLMClient


class BaseValidationAgent(ABC):

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self.role = self.__class__.__name__
        self.logger = logging.getLogger(f"marva.s3.agent.{self.role}")

    @abstractmethod
    def run(self, input_data: dict) -> dict[str, Any]:
        raise NotImplementedError
