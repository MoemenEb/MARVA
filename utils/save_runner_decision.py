from pathlib import Path
from datetime import datetime
import json

def save_runner_decision(decision_summary: dict, path:str):
    decision_out_dir = Path(path / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    decision_out_dir.mkdir(parents=True, exist_ok=True)
    with open(decision_out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(decision_summary, f, indent=2, ensure_ascii=False)