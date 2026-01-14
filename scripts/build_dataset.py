import csv
import json
from pathlib import Path

# ----------------------------
# Paths
# ----------------------------
RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Dataset registry
# ----------------------------
DATASETS = {
    "qure": "QuRE_raw.csv",
    "promised": "promised_raw.csv",
    "leeds": "leed_raw.csv",
    "dronology": "dronology_raw.csv",
    "reqview": "reqview_raw.csv",
    "wasp": "wasp_raw.csv",
    "synt": "syntatic_raw.csv"
}

# ----------------------------
# Build canonical dataset
# ----------------------------
all_requirements = []
global_index = 1

for source, filename in DATASETS.items():
    csv_path = RAW_DIR / filename

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing dataset file: {csv_path}")

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if "id" not in reader.fieldnames or "requirement" not in reader.fieldnames:
            raise ValueError(
                f"{filename} must contain exactly 'id' and 'requirement' columns"
            )

        for row in reader:
            text = row["requirement"].strip()
            if not text:
                continue  # skip empty rows

            record = {
                "req_id": f"{source.upper()}-{global_index:06d}",
                "text": text,
                "source": source,
                "group_id": None,
                "metadata": {
                    "original_id": row["id"].strip()
                }
            }

            all_requirements.append(record)
            global_index += 1

# ----------------------------
# Deterministic ordering
# ----------------------------
all_requirements.sort(key=lambda r: r["req_id"])

# ----------------------------
# Write output
# ----------------------------
output_path = OUT_DIR / "requirements.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(all_requirements, f, indent=2, ensure_ascii=False)

print(f"[OK] Saved {len(all_requirements)} requirements to {output_path}")
