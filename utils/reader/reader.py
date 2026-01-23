import os
from .csv_reader import CSVReader
from .txt_reader import TXTReader
from .base_reader import BaseReader


class Reader:

    def get_reader(file_path: str) -> BaseReader:
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        if ext == ".csv":
            return CSVReader()

        if ext == ".txt":
            return TXTReader()

        raise NotImplementedError(
            f"Unsupported file type: {ext}"
        )
