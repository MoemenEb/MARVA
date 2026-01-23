from datetime import datetime
import json
import argparse
import logging
from pathlib import Path

from common.llm_client import LLMClient
from s2.validation_agents import ValidatorAgent
from s2.validation_summary import ValidatorSummary
from utils.reader.reader import Reader
from entity.requirement_set import RequirementSet
from common.logging.setup import setup_logging
from s2.logger import init_s2_logger


DATA_PATH = Path("data/processed/requirements_grouped.json")
DECISON_OUTPUT_PATH = Path("out/s2_decisions/")
RAW_DATA_PATH = Path("data/raw/")



def main(mode: str, scope: str, limit: int | None):

    setup_logging(run_id="s2_run_"+datetime.now().strftime('%Y%m%d'))
    init_s2_logger()
    logger = logging.getLogger("marva.s2.runner_new")
    # -----------------------------
    # Load dataset
    # -----------------------------
    DATA_PATH = RAW_DATA_PATH / f"{scope}"
    reader = Reader.get_reader(DATA_PATH)
    requirements_set = reader.read(DATA_PATH)
    if limit:
        requirements = requirements_set[:limit]
    requirementSet = RequirementSet(requirements)
    logger.info(f"Starting S2 runner with mode={mode}, scope={scope}, limit={limit}")

    # -----------------------------
    # Grouping (needed for group scope)
    # -----------------------------
    groups = requirementSet

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


    # agent = S2ValidatorAgent(llm)
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
            out_file = out_dir / f"{req.id}.json"
            # if out_file.exists():
            #     continue

            # group = groups.get(req.group_id)
            # result = agent.run(req, group, mode)
            result = agents.execute(mode=mode, requirement=req, group=None)
            print("result:", result)
            summary = summarizer.summarize(result, req)
            print("Summary:", summary)

            final_decision["flow_latency_seconds"] = (_calculate_total_latency(result) + summary["latency_ms"])/1000

            decision = {
                "requirement_id": req.id,
                "requirement_text": req.text,
                "decision": summary["output"]["final_status"],
                "by_agent" : {k: v["output"]["decision"] for k, v in result.items()},
                "recommendations": summary["output"]["recommendations"],
            }
            final_decision["Validation Decisions"].append(decision)

            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            print(f"[S2-single] {req.id} done")

        with open(decision_out_dir / "decision_summary.json", "w", encoding="utf-8") as f:
            json.dump(final_decision, f, indent=2, ensure_ascii=False)


    elif mode == "group":
        reqi = {"requirements": []}
        out_file = out_dir / f"group_{req.id}.json"
        for req in requirementSet.requirements:
            requir = {
                    'requirement_id': req.id,
                    'requirement_text': req.text
                }
            reqi["requirements"].append(requir)
    
        print("Executing group validation...")
        result = agents.execute(mode=mode, requirement=None, group=requirementSet)
        print("Group result:", result)
        summary = summarizer.summarize(result, reqi)
        print("Group summary:", summary)
        final_decision["flow_latency_seconds"] = (_calculate_total_latency(result) + summary["latency_ms"])/1000
        decision = {
            **reqi,
            "decision": summary["output"]["final_status"],
            "by_agent" : {k: v["output"]["decision"] for k, v in result.items()},
            "recommendations": summary["output"]["recommendations"],
        }
        final_decision["Validation Decisions"].append(decision)

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"[S2-group] {req.id} done")
            
        with open(decision_out_dir / "decision_summary.json", "w", encoding="utf-8") as f:
            json.dump(final_decision, f, indent=2, ensure_ascii=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run S2 baseline")
    parser.add_argument("--scope", required=True)
    parser.add_argument("--mode", required=True, choices=["single", "group"])
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()
    main(args.mode, args.scope, args.limit)
