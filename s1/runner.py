import json
import argparse
from pathlib import Path

from common.llm_client import LLMClient
from common.dataset_selector import filter_requirements
from s1.pipeline import S1Pipeline

DATA_PATH = Path("data/processed/requirements.json")

def main(mode: str, limit: int | None):
    with open(DATA_PATH, encoding="utf-8") as f:
        requirements = json.load(f)

    requirements = filter_requirements(requirements, mode)

 # --- Apply limit (DEBUG ONLY) ---
    if limit is not None:
        requirements = requirements[:limit]


    llm = LLMClient(
        host="http://localhost:11434",
        model="llama3.2"
    )

    pipeline = S1Pipeline(llm)

    out_dir = Path(f"s1/outputs/{mode}")
    out_dir.mkdir(parents=True, exist_ok=True)

    for req in requirements:
        out_file = out_dir / f"{req['req_id']}.json"
        if out_file.exists():
            continue  # resume-safe

        result = pipeline.run(req)

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"[S1] {req['req_id']} done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True,
                        help="Execution mode: small | large | all | dataset name")
    parser.add_argument("--limit", type=int, default=None,
                    help="Max number of requirements to process")

    args = parser.parse_args()

    main(args.mode, args.limit)
