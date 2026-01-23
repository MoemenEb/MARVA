import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | "
    "run=%(run_id)s | %(message)s"
)

class RunIdFilter(logging.Filter):
    def __init__(self, run_id: str):
        super().__init__()
        self.run_id = run_id

    def filter(self, record):
        record.run_id = self.run_id
        return True


def setup_logging(run_id: str, level=logging.INFO):
    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(LOG_FORMAT)

    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.addFilter(RunIdFilter(run_id))

    # Global audit file
    file = RotatingFileHandler(
        LOG_DIR / "marva.log",
        maxBytes=5_000_000,
        backupCount=5
    )
    file.setFormatter(formatter)
    file.addFilter(RunIdFilter(run_id))

    root.handlers.clear()
    root.addHandler(console)
    root.addHandler(file)
