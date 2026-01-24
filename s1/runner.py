from datetime import datetime
import argparse
import time
import logging
from pathlib import Path
from utils.dataset_loader import load_dataset
from utils.save_runner_decision import save_runner_decision

from common.llm_client import LLMClient
from s1.pipeline import S1Pipeline
from common.logging.setup import setup_logging
from s1.logger import init_s1_logger

DECISON_OUTPUT_PATH = Path("out/s1_decisions/")
LOGGER = "marva.s1.runner"



def main(mode: str, scope: str, limit: int | None):
    setup_logging(run_id="s1_run_"+datetime.now().strftime('%Y%m%d'))
    init_s1_logger()
    logger = logging.getLogger(LOGGER)
    logger.info(f"Starting S1 runner with mode={mode}, scope={scope}, limit={limit}")
    
    logger.info(f"Loading dataset {scope}")
    requirement_set = load_dataset(scope, limit)
    logger.info(f"Loaded {len(requirement_set.requirements)} requirements from dataset")

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

    pipeline = S1Pipeline(llm)
    decision_summary = {
        "mode": mode,
        "scope": scope,
        "Validation framework" : "S1 Validation Agent v1.0",
        "flow_latency_seconds" : 0,
        "validation_decision": [],
    }

    startTime = time.perf_counter()
    # -----------------------------
    # SINGLE SCOPE
    # -----------------------------
    if mode == "single":
        for req in requirement_set.requirements:    
            json_result = pipeline.execute(req, mode)

            summary = {
                "requirements": {
                    req.id: req.text
                },
                **json_result,
            }
            logger.debug(f"Validation summary for requirement ID {req.id}: {summary}")
            logger.info(f"[S1|single] {req.id} done")
            decision_summary["validation_decision"].append(summary)
    # -----------------------------
    # GROUP SCOPE
    # -----------------------------
    elif mode == "group":
        json_result = pipeline.execute(requirement_set.requirements, mode)
        summary = {
            "requirements": {
                r.id: r.text
                for r in requirement_set.requirements
            },
            **json_result,
        }
        logger.debug(f"Validation summary for group: {summary}")
        logger.info("[S1|group] group analysis done")
        decision_summary["validation_decision"].append(summary)
    else:
        raise ValueError(f"Invalid scope: {scope}")

    endTime = time.perf_counter()
    flowlatency = int((endTime - startTime))
    logger.info(f"S1 runner completed in {flowlatency} seconds")
    decision_summary["flow_latency_seconds"] = flowlatency
    
    save_runner_decision(decision_summary, DECISON_OUTPUT_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scope",
        required=True,
        help="dataset name",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of requirements to process",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["single", "group"],
        help="S1 execution scope",
    )

    args = parser.parse_args()

    main(args.mode, args.scope, args.limit)
