import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("marva.save_decision")


def save_runner_decision(decision_summary: dict, path: Path) -> Path:
    decision_out_dir = Path(path / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    decision_out_dir.mkdir(parents=True, exist_ok=True)
    output_file = decision_out_dir / "summary.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(decision_summary, f, indent=2, ensure_ascii=False)
    logger.debug("Decision saved to %s", output_file)
    return decision_out_dir
