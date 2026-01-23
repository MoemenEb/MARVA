import re
from typing import List
from .base_reader import BaseReader
from entity.requirement import Requirement


class TXTReader(BaseReader):

    SENTENCE_SPLIT_REGEX = r'(?<=[.!?])\s+'

    def read(self, file_path: str) -> List[Requirement]:
        requirements: List[Requirement] = []

        with open(file_path, encoding="utf-8") as f:
            content = f.read().strip()

        if not content:
            return requirements

        sentences = re.split(self.SENTENCE_SPLIT_REGEX, content)

        for idx, sentence in enumerate(sentences, start=1):
            sentence = sentence.strip()

            if not sentence:
                continue

            requirements.append(
                Requirement(
                    req_id=f"T{idx}",
                    text=sentence
                )
            )

        return requirements
