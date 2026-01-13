from datetime import datetime
import json
import argparse
from pathlib import Path
import time

from common.dataset_selector import filter_requirements
from s2.grouping import group_requirements

from s3.graph import build_marva_s3_graph
from s3.agents import build_stub_agents

DATA_PATH = Path("data/processed/requirements_grouped.json")
DECISON_OUTPUT_PATH = Path("out/s3_decisions/")


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

    requirements = filter_requirements(requirements, scope)

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

    out_dir = Path(f"s3/outputs/{scope}/{mode}")
    out_dir.mkdir(parents=True, exist_ok=True)

    decision_out_dir = Path(DECISON_OUTPUT_PATH / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    decision_out_dir.mkdir(parents=True, exist_ok=True)

    decision_summary = {
        "mode": mode,
        "scope": scope,
        "validation framework" : "MARVA v1.0",
        "flow_latency_seconds": None
    }

    # -----------------------------
    # Execute (SEMANTIC PARITY WITH S2)
    # -----------------------------
    if mode == "single":
        startTime = time.perf_counter()
        single_out = {
            "Validation Decisions": []
        }
        for req in requirements:
            out_file = out_dir / f"{req['req_id']}.json"
            # if out_file.exists():
            #     continue

            state = {
                "mode": "single",
                "requirement": req,
            }
            result = app.invoke(state)
            decision = result["decision"]

            full = {
                "requirement_id": req['req_id'],
                "requirement_text": req['text'],
                **decision
            }
            single_out["Validation Decisions"].append(full)

            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            print(f"[S3-single] {req['req_id']} done")
        endTime = time.perf_counter()
        flowlatency = int((endTime - startTime))   
        decision_summary["flow_latency_seconds"] = flowlatency 
        final_decision = {
            **decision_summary,
            **single_out,
        }
        with open(decision_out_dir / "decision_summary.json", "w", encoding="utf-8") as f:
            json.dump(final_decision, f, indent=2, ensure_ascii=False)

    elif mode == "group":
        startTime = time.perf_counter()
        for group_id, group_reqs in groups.items():
            if group_id is None:
                continue  # same behavior as S2

            out_file = out_dir / f"group_{group_id}_run{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            state = {
                "mode": "group",
                "group": group_reqs,
            }

            reqi = {"requirements": []}

            for req in group_reqs:
                req_ids = req.get('req_id', 'unknown')
                req_texts = req.get('text', 'unknown')
                requir = {
                    'req_id': req_ids,
                    'text': req_texts
                }
                reqi["requirements"].append(requir)
                

            result = app.invoke(state)
            decision = result["decision"]

            full = {
                "group_id": group_id,
                **reqi,
                **decision
            }
            endTime = time.perf_counter()
            flowlatency = int((endTime - startTime))
            decision_summary["flow_latency_seconds"] = flowlatency
            final_decision = {
            **decision_summary,
            **full,
            }
            with open(decision_out_dir / "decision_summary.json", "w", encoding="utf-8") as f:
                json.dump(final_decision, f, indent=2, ensure_ascii=False)

            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            print(f"[S3-group] {group_id} done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run S3 (MARVA)")
    parser.add_argument("--scope", required=True)
    parser.add_argument("--mode", required=True, choices=["single", "group"])
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()
    main(args.mode, args.scope, args.limit)