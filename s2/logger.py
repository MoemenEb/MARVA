import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

def init_s2_logger():
    logger = logging.getLogger("marva.s2")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return

    handler = RotatingFileHandler(
        Path("logs/s2.log"),
        maxBytes=3_000_000,
        backupCount=3
    )

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | S2 | %(name)s | %(message)s"
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = True
