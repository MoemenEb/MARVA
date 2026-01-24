from entity.requirement_set import RequirementSet
from utils.reader.reader import Reader
from pathlib import Path

RAW_DATA_PATH = Path("data/raw/")


def load_dataset(path: str, limit: int | None = None) -> RequirementSet:
    DATA_PATH = RAW_DATA_PATH / f"{path}"
    reader = Reader.get_reader(DATA_PATH)
    requirements_set = reader.read(DATA_PATH)
    if limit:
        requirements = requirements_set[:limit]
    requirementSet = RequirementSet(requirements)
    return requirementSet