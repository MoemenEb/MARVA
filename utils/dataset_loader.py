from entity.requirement_set import RequirementSet
from utils.reader.reader import Reader
from pathlib import Path

RAW_DATA_PATH = Path("data/raw/")


def load_dataset(path: str, limit: int | None = None) -> RequirementSet:
    data_path = RAW_DATA_PATH / f"{path}"
    reader = Reader.get_reader(data_path)
    requirements = reader.read(data_path)
    if limit:
        requirements = requirements[:limit]
    return RequirementSet(requirements)