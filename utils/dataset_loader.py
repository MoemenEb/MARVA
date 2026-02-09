import logging
import time
from entity.requirement_set import RequirementSet
from utils.reader.reader import Reader
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data"
logger = logging.getLogger("marva.dataset_loader")


def load_dataset(path: str, limit: int | None = None) -> RequirementSet:
    data_path = DATA_PATH / f"{path}"
    logger.debug("Loading dataset from %s", data_path)
    t0 = time.perf_counter()
    reader = Reader.get_reader(data_path)
    requirements = reader.read(data_path)
    read_elapsed = time.perf_counter() - t0
    total = len(requirements)
    if limit:
        requirements = requirements[:limit]
        logger.debug("Applied limit: %d/%d requirements", len(requirements), total)
    logger.debug("Dataset loaded: %d requirements in %.2fs", len(requirements), read_elapsed)
    return RequirementSet(requirements)
