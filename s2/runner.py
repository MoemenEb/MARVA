from datetime import datetime
import json
import argparse
from pathlib import Path

from common.dataset_selector import filter_requirements
from common.llm_client import LLMClient
from s2.grouping import group_requirements
from s2.validator_agent import S2ValidatorAgent

DATA_PATH = Path("data/processed/requirements_grouped.json")
DECISON_OUTPUT_PATH = Path("out/s2_decisions/")


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
        model="qwen3:1.7b",
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

    out_dir = Path(f"s2/outputs/{scope}/{mode}")
    out_dir.mkdir(parents=True, exist_ok=True)

    decision_out_dir = Path(DECISON_OUTPUT_PATH / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    decision_out_dir.mkdir(parents=True, exist_ok=True)


    # -----------------------------
    # Execute
    # -----------------------------
    if mode == "single":
        final_decision = {
            "mode": "single",
            "scope": scope,
            "Validation Decisions": [],
            "latency": 0
        }
        for req in requirements:
            out_file = out_dir / f"{req['req_id']}.json"
            # if out_file.exists():
            #     continue

            group = groups.get(req["group_id"])
            result = agent.run(req, group, scope)
            
            final_decision["latency"] += result["flow_latency_seconds"]
            
            decision = {
                "req_id": req["req_id"],
                "req_text": req["text"],
                "agent" : "S2 Validator Agent",
                "decision": result["summary"]["output"]["final_status"],
                "by_agent" : {k: v["output"]["decision"] for k, v in result["results"].items()},
                "recommendations": result["summary"]["output"]["recommendations"],
            }
            final_decision["Validation Decisions"].append(decision)

            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            print(f"[S2-single] {req['req_id']} done")

        with open(decision_out_dir / "decision_summary.json", "w", encoding="utf-8") as f:
            json.dump(final_decision, f, indent=2, ensure_ascii=False)


    elif mode == "group":
        final_decision = {
            "mode": "group",
            "scope": scope,
            "Validation Decisions": [],
            "latency": 0
        }
        for group_id, group_reqs in groups.items():
            if group_id is None:
                continue  # skip ungrouped if needed

            out_file = out_dir / f"group_{group_id}.json"
            # if out_file.exists():
            #     continue
            reqi = {"requirements": []}

            for req in group_reqs:
                req_ids = req.get('req_id', 'unknown')
                req_texts = req.get('text', 'unknown')
                requir = {
                    'req_id': req_ids,
                    'text': req_texts
                }
                reqi["requirements"].append(requir)
                
            # One execution per group
            result = agent.run(None, group_reqs, scope)
            final_decision["latency"] += result["flow_latency_seconds"]
            decision = {
                **reqi,
                "agent" : "S2 Validator Agent",
                "decision": result["summary"]["output"]["final_status"],
                "by_agent" : {k: v["output"]["decision"] for k, v in result["results"].items()},
                "recommendations": result["summary"]["output"]["recommendations"],
            }
            final_decision["Validation Decisions"].append(decision)

            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            print(f"[S2-group] {group_id} done")
            
        with open(decision_out_dir / "decision_summary.json", "w", encoding="utf-8") as f:
            json.dump(final_decision, f, indent=2, ensure_ascii=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run S2 baseline")
    parser.add_argument("--scope", required=True)
    parser.add_argument("--mode", required=True, choices=["single", "group"])
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()
    main(args.mode, args.scope, args.limit)
