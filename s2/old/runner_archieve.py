from datetime import datetime
import json
import argparse
from pathlib import Path

from common.dataset_selector import filter_requirements
from common.llm_client import LLMClient
from s2.old.grouping import group_requirements
from s2.validator_agent import S2ValidatorAgent
from s2.validation_agents import ValidatorAgent
from s2.validation_summary import ValidatorSummary

DATA_PATH = Path("data/processed/requirements_grouped.json")
DECISON_OUTPUT_PATH = Path("out/s2_decisions/")


def main(mode: str, scope: str, limit: int | None):
    # -----------------------------
    # Load dataset
    # -----------------------------
    with open(DATA_PATH, encoding="utf-8") as f:
        requirements = json.load(f)

    requirements = filter_requirements(requirements, scope)

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
    agents = ValidatorAgent(llm)
    summarizer = ValidatorSummary(llm)

    out_dir = Path(f"s2/outputs/{scope}/{mode}")
    out_dir.mkdir(parents=True, exist_ok=True)

    decision_out_dir = Path(DECISON_OUTPUT_PATH / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    decision_out_dir.mkdir(parents=True, exist_ok=True)

    def _calculate_total_latency(results: dict) -> int:
        """Calculate total latency from validation results."""
        print(results)
        return sum(
            item['latency_ms'] 
            for item in results.values() 
            if isinstance(item, dict) and 'latency_ms' in item
        )
    
    final_decision = {
            "mode": mode,
            "scope": scope,
            "Validation framework" : "S2 Validation Agent v1.0",
            "flow_latency_seconds": 0,
            "Validation Decisions": [],
        }

    # -----------------------------
    # Execute
    # -----------------------------
    if mode == "single":
        for req in requirements:
            out_file = out_dir / f"{req['req_id']}.json"
            # if out_file.exists():
            #     continue

            group = groups.get(req["group_id"])
            # result = agent.run(req, group, mode)
            result = agents.execute(mode=mode, requirement=req, group=None)
            # print("result:", result)
            summary = summarizer.summarize(result, req)
            # print("summary:", summary)
            final_decision["flow_latency_seconds"] = (_calculate_total_latency(result) + summary["latency_ms"])/1000

            decision = {
                "requirement_id": req["req_id"],
                "requirement_text": req["text"],
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
                    'requirement_id': req_ids,
                    'requirement_text': req_texts
                }
                reqi["requirements"].append(requir)
                
            # One execution per group
            # result = agent.run(None, group_reqs, mode)
            result = agents.execute(mode=mode, requirement=None, group=group_reqs)
            summary = summarizer.summarize(result, group_reqs)
            final_decision["flow_latency_seconds"] = result["latency_ms"]/1000 + summary["latency_ms"]/1000
            decision = {
                **reqi,
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
