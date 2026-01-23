from abc import ABC, abstractmethod
from typing import List
from entity.requirement import Requirement


class BaseReader(ABC):

    @abstractmethod
    def read(self, file_path: str) -> List[Requirement]:
        """
        Read input file and return a list of Requirement objects.
        Readers must NOT perform any validation.
        """
        pass
