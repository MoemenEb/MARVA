import json
import argparse
from pathlib import Path

from common.llm_client import LLMClient
from common.dataset_selector import filter_requirements
from s1.pipeline import S1Pipeline

DATA_PATH = Path("data/processed/requirements.json")

OUT_DIR = Path("s1/outputs/{mode}/{scope}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main(mode: str, scope: str, limit: int | None):
    with open(DATA_PATH, encoding="utf-8") as f:
        requirements = json.load(f)

    requirements = filter_requirements(requirements, mode)

    if limit:
        requirements = requirements[:limit]

    llm = LLMClient(
        host="http://localhost:11434",
        model="deepseek-r1:1.5b"
    )
    #gemma3n:e2b
    #gemma3:4b
    #deepseek-r1:1.5b
    #qwen3:1.7b
    #qwen3:4b (heavy)
    #llama3.2 (flaky)
    #falcon3:3b
    #ministral-3:3b

    pipeline = S1Pipeline(llm)

    # -----------------------------
    # SINGLE SCOPE
    # -----------------------------
    if scope == "single":
        for req in requirements:
            result = pipeline.run_single(req)
            # OUT_DIR = Path("s1/output/single")
            # OUT_DIR.mkdir(parents=True, exist_ok=True)
            out_file = OUT_DIR / f"{req['req_id']}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            print(f"[S1|single] {req['req_id']} done")

    # -----------------------------
    # GROUP SCOPE
    # -----------------------------
    elif scope == "group":
        result = pipeline.run_group(requirements)
        # OUT_DIR = Path("s1/output/group")
        # OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_file = OUT_DIR / "s1_group.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print("[S1|group] group analysis done")

    else:
        raise ValueError(f"Invalid scope: {scope}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        help="Execution mode: small | large | all | dataset name",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of requirements to process",
    )
    parser.add_argument(
        "--scope",
        required=True,
        choices=["single", "group"],
        help="S1 execution scope",
    )

    args = parser.parse_args()

    main(args.mode, args.scope, args.limit)
