import csv
import json
import logging
from pathlib import Path

logger = logging.getLogger("marva.save_csv")


def _format_recommendations(recommendations) -> str:
    if recommendations is None:
        return ""
    if isinstance(recommendations, list):
        return " | ".join(str(item) for item in recommendations)
    if isinstance(recommendations, dict):
        return json.dumps(recommendations, ensure_ascii=False)
    return str(recommendations)


def _validation_status_map(validations) -> dict:
    mapping = {}
    for validation in validations or []:
        agent = str(validation.get("agent", "")).strip().lower()
        if not agent:
            continue
        if agent.endswith("_group"):
            agent = agent[: -len("_group")]
        elif agent.endswith("_single"):
            agent = agent[: -len("_single")]
        mapping[agent] = validation.get("status", "")
    return mapping


def save_runner_csv(requirement_set, mode: str, duration_seconds: float, output_dir: Path) -> Path:
    output_file = output_dir / "results.csv"

    if mode == "single":
        headers = [
            "id",
            "requirement",
            "atomicity",
            "clarity",
            "completion",
            "final_decision",
            "recommendations",
            "duration",
        ]
        rows = []
        for req in requirement_set.requirements:
            validation_map = _validation_status_map(req.single_validations)
            rows.append(
                {
                    "id": req.id,
                    "requirement": req.text,
                    "atomicity": validation_map.get("atomicity", ""),
                    "clarity": validation_map.get("clarity", ""),
                    "completion": validation_map.get("completion", ""),
                    "final_decision": req.final_decision,
                    "recommendations": _format_recommendations(req.recommendation),
                    "duration": req.duration_seconds,
                }
            )
    else:
        headers = [
            "id",
            "requirement",
            "redundancy",
            "completion",
            "consistency",
            "final_decision",
            "recommendations",
            "duration",
        ]
        rows = []
        validation_map = _validation_status_map(requirement_set.group_validations)
        for idx, req in enumerate(requirement_set.requirements):
            if idx == 0:
                redundancy = validation_map.get("redundancy", "")
                completion = validation_map.get("completion", "")
                consistency = validation_map.get("consistency", "")
                final_decision = requirement_set.final_decision
                recommendations = _format_recommendations(requirement_set.recommendations)
                duration = duration_seconds
            else:
                redundancy = ""
                completion = ""
                consistency = ""
                final_decision = ""
                recommendations = ""
                duration = ""
            rows.append(
                {
                    "id": req.id,
                    "requirement": req.text,
                    "redundancy": redundancy,
                    "completion": completion,
                    "consistency": consistency,
                    "final_decision": final_decision,
                    "recommendations": recommendations,
                    "duration": duration,
                }
            )

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    logger.debug("CSV results saved to %s", output_file)
    return output_file
