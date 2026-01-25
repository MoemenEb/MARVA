from datetime import datetime
import argparse
import logging
from pathlib import Path

from common.llm_client import LLMClient
from s2.validation_agents import ValidatorAgent
from s2.validation_summary import ValidatorSummary
from utils.dataset_loader import load_dataset
from common.logging.setup import setup_logging
from s2.logger import init_s2_logger
from utils.save_runner_decision import save_runner_decision


DECISON_OUTPUT_PATH = Path("out/s2_decisions/")



def main(mode: str, scope: str, limit: int | None):

    setup_logging(run_id="s2_run_"+datetime.now().strftime('%Y%m%d'))
    init_s2_logger()
    logger = logging.getLogger("marva.s2.runner_new")
    # -----------------------------
    # Load dataset
    # -----------------------------
    logger.info(f"Loading dataset {scope}")
    requirement_set = load_dataset(scope, limit)
    logger.info(f"Loaded {len(requirement_set.requirements)} requirements from dataset")

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


    agents = ValidatorAgent(llm)
    summarizer = ValidatorSummary(llm)

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
        for req in requirement_set.requirements:
            result = agents.execute(mode=mode, requirement=req, group=None)
            summary = summarizer.summarize(result, req)
            final_decision["flow_latency_seconds"] = (_calculate_total_latency(result) + summary["latency_ms"])/1000

            decision = {
                "requirement_id": req.id,
                "requirement_text": req.text,
                "decision": summary["output"]["final_status"],
                "by_agent" : {k: v["output"]["decision"] for k, v in result.items()},
                "recommendations": summary["output"]["recommendations"],
            }
            final_decision["Validation Decisions"].append(decision)
            logger.info(f"[S2-single] {req.id} done")


    elif mode == "group":
        reqi = {"requirements": []}
        for req in requirement_set.requirements:
            requir = {
                    'requirement_id': req.id,
                    'requirement_text': req.text
                }
            reqi["requirements"].append(requir)
    
        print("Executing group validation...")
        result = agents.execute(mode=mode, requirement=None, group=requirement_set)
        summary = summarizer.summarize(result, reqi)
        final_decision["flow_latency_seconds"] = (_calculate_total_latency(result) + summary["latency_ms"])/1000
        decision = {
            **reqi,
            "decision": summary["output"]["final_status"],
            "by_agent" : {k: v["output"]["decision"] for k, v in result.items()},
            "recommendations": summary["output"]["recommendations"],
        }
        final_decision["Validation Decisions"].append(decision)
        logger.info(f"[S2-group] {req.id} done")

    save_runner_decision(final_decision, DECISON_OUTPUT_PATH)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run S2 baseline")
    parser.add_argument("--scope", required=True)
    parser.add_argument("--mode", required=True, choices=["single", "group"])
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()
    main(args.mode, args.scope, args.limit)
