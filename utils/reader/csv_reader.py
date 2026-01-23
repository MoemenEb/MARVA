import csv
from typing import List
from .base_reader import BaseReader
from entity.requirement import Requirement


class CSVReader(BaseReader):

    def read(self, file_path: str) -> List[Requirement]:
        requirements: List[Requirement] = []

        with open(file_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            if not reader.fieldnames:
                raise ValueError("CSV file has no header")

            if "id" not in reader.fieldnames or "requirement" not in reader.fieldnames:
                raise ValueError("CSV must contain 'id' and 'requirement' columns")

            for row in reader:
                req_id = row["id"].strip()
                text = row["requirement"].strip()

                if not req_id or not text:
                    # silently skip empty rows
                    continue

                requirements.append(
                    Requirement(
                        req_id=req_id,
                        text=text
                    )
                )

        return requirements
