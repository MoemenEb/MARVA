import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

def init_s1_logger():
    logger = logging.getLogger("marva.s1")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return

    handler = RotatingFileHandler(
        Path("logs/s1.log"),
        maxBytes=3_000_000,
        backupCount=3
    )

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | S1 | %(name)s | %(message)s"
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = True
