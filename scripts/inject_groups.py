import json
import re
from pathlib import Path

IN_PATH = Path("data/processed/requirements.json")
OUT_PATH = Path("data/processed/requirements_grouped.json")

if not IN_PATH.exists():
    raise FileNotFoundError(f"Missing input file: {IN_PATH}")

with open(IN_PATH, encoding="utf-8") as f:
    requirements = json.load(f)

grouped = []

for r in requirements:
    source = r["source"]
    original_id = r["metadata"]["original_id"]

    group_id = None

    # --- PROMISED ---
    if source == "promised":
        match = re.match(r"(Prj-\d+)", original_id)
        if not match:
            raise ValueError(f"Cannot extract project ID from PROMISED id: {original_id}")
        project = match.group(1)
        group_id = f"{source.upper()}-{project}"

    # --- OTHER GROUPED DATASETS ---
    elif source in {"leeds", "dronology", "reqview", "wasp"}:
        if "-" not in original_id:
            raise ValueError(f"Expected '-' in original_id for {source}: {original_id}")
        project = original_id.split("-", 1)[0].strip()
        group_id = f"{source.upper()}-{project}"

    # --- QURE (NO GROUPING) ---
    elif source == "qure":
        group_id = None

    else:
        raise ValueError(f"Unknown dataset source: {source}")

    r_out = dict(r)
    r_out["group_id"] = group_id
    grouped.append(r_out)

# Deterministic ordering
grouped.sort(key=lambda x: x["req_id"])

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(grouped, f, indent=2, ensure_ascii=False)

print(f"[OK] Grouped dataset written to {OUT_PATH}")
