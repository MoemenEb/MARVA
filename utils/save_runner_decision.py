import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("marva.save_decision")


def save_runner_decision(decision_summary: dict, path: Path) -> Path:
    decision_out_dir = Path(path / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    decision_out_dir.mkdir(parents=True, exist_ok=True)
    mode = str(decision_summary.get("Mode", "")).lower()

    if mode == "single":
        validation_list = decision_summary.get("Validation", [])
        summary_validation = [
            {
                "id": req.get("id"),
                "text": req.get("text"),
                "final_decision": req.get("final_decision"),
                "recommendation": req.get("recommendation"),
            }
            for req in validation_list
        ]
        summary_payload = {
            "Framework": decision_summary.get("Framework"),
            "Mode": decision_summary.get("Mode"),
            "Duration": decision_summary.get("Duration"),
            "Validation": summary_validation,
        }
        detailed_validation = [
            {k: v for k, v in req.items() if k != "final_decision"}
            for req in validation_list
        ]
        detailed_payload = {
            "Framework": decision_summary.get("Framework"),
            "Mode": decision_summary.get("Mode"),
            "Duration": decision_summary.get("Duration"),
            "Validation": detailed_validation,
        }
    else:
        summary_payload = decision_summary
        detailed_payload = None

    summary_file = decision_out_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)
    logger.debug("Decision summary saved to %s", summary_file)

    if detailed_payload is not None:
        detailed_file = decision_out_dir / "detailed.json"
        with open(detailed_file, "w", encoding="utf-8") as f:
            json.dump(detailed_payload, f, indent=2, ensure_ascii=False)
        logger.debug("Decision details saved to %s", detailed_file)
    return decision_out_dir
