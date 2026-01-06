import json
import argparse
from pathlib import Path

from common.dataset_selector import filter_requirements
from common.llm_client import LLMClient
from s2.grouping import group_requirements
from s2.validator_agent import S2ValidatorAgent

DATA_PATH = Path("data/processed/requirements_grouped.json")


def main(mode: str, scope: str, limit: int | None):
    # -----------------------------
    # Load dataset
    # -----------------------------
    with open(DATA_PATH, encoding="utf-8") as f:
        requirements = json.load(f)

    requirements = filter_requirements(requirements, mode)

    if limit is not None:
        requirements = requirements[:limit]

    print(f"[S2] mode={mode}, scope={scope}, count={len(requirements)}")

    # -----------------------------
    # Grouping (needed for group scope)
    # -----------------------------
    groups = group_requirements(requirements)

    # -----------------------------
    # Init LLM + agent
    # -----------------------------
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


    agent = S2ValidatorAgent(llm)

    out_dir = Path(f"s2/outputs/{mode}/{scope}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Execute
    # -----------------------------
    if scope == "single":
        for req in requirements:
            out_file = out_dir / f"{req['req_id']}.json"
            if out_file.exists():
                continue

            group = groups.get(req["group_id"])
            result = agent.run(req, group, scope)

            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            print(f"[S2-single] {req['req_id']} done")


    elif scope == "group":
        for group_id, group_reqs in groups.items():
            if group_id is None:
                continue  # skip ungrouped if needed

            out_file = out_dir / f"group_{group_id}.json"
            if out_file.exists():
                continue

            # One execution per group
            result = agent.run(None, group_reqs, scope)

            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            print(f"[S2-group] {group_id} done")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run S2 baseline")
    parser.add_argument("--mode", required=True)
    parser.add_argument("--scope", required=True, choices=["single", "group"])
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()
    main(args.mode, args.scope, args.limit)
