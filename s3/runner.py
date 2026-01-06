import json
import argparse
from pathlib import Path

from common.dataset_selector import filter_requirements
from s2.grouping import group_requirements

from s3.graph import build_marva_s3_graph
from s3.agents import build_stub_agents

DATA_PATH = Path("data/processed/requirements_grouped.json")


def main(mode: str, scope: str, limit: int | None):
    """
    S3 runner (MARVA)

    mode: dataset selector (wasp, reqview, ...)
    scope: single | group
    """

    # -----------------------------
    # Load dataset (IDENTICAL to S2)
    # -----------------------------
    with open(DATA_PATH, encoding="utf-8") as f:
        requirements = json.load(f)

    requirements = filter_requirements(requirements, mode)

    if limit is not None:
        requirements = requirements[:limit]

    print(f"[S3] mode={mode}, scope={scope}, count={len(requirements)}")

    # -----------------------------
    # Grouping (IDENTICAL to S2)
    # -----------------------------
    groups = group_requirements(requirements)

    # -----------------------------
    # Init agents + graph (S3-specific)
    # -----------------------------
    agents = build_stub_agents()
    graph = build_marva_s3_graph(agents)
    app = graph.compile()

    out_dir = Path(f"s3/outputs/{mode}/{scope}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Execute (SEMANTIC PARITY WITH S2)
    # -----------------------------
    if scope == "single":
        for req in requirements:
            out_file = out_dir / f"{req['req_id']}.json"
            if out_file.exists():
                continue

            state = {
                "mode": "single",
                "requirement": req,
                "group": [req],  # placeholder, unused in single mode
            }
            result = app.invoke(state)

            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            print(f"[S3-single] {req['req_id']} done")

    elif scope == "group":
        for group_id, group_reqs in groups.items():
            if group_id is None:
                continue  # same behavior as S2

            out_file = out_dir / f"group_{group_id}.json"
            if out_file.exists():
                continue

            state = {
                "mode": "group",
                "requirement": group_reqs[0],  # required by schema
                "group": group_reqs,
            }

            result = app.invoke(state)

            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            print(f"[S3-group] {group_id} done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run S3 (MARVA)")
    parser.add_argument("--mode", required=True)
    parser.add_argument("--scope", required=True, choices=["single", "group"])
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()
    main(args.mode, args.scope, args.limit)