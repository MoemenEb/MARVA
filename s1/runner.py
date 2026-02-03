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

from entity.decision import Decision

DECISION_OUTPUT_PATH = Path("out/s1_decisions/")
LOGGER = "marva.s1.runner"
FRAMEWORK = "S1 Validation Agent v1.0"



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

    pipeline = S1Pipeline(llm)
    decision = Decision(
        framework=FRAMEWORK,
        mode=mode  
    )

    start_time = time.perf_counter()
    logger.info("Start S1 pipeline")
    pipeline.run(requirement_set, mode)
    logger.info("S1 pipeline finished")
    logger.info("Saving results ...")
    
    dec = (
            [req.to_dict() for req in requirement_set.requirements]
            if mode == "single"
            else requirement_set.to_dict()
        )
    
    decision.duration = int((time.perf_counter() - start_time))
    decision.decision = dec
    
    output_dir = save_runner_decision(decision.to_dict(), DECISION_OUTPUT_PATH)
    logger.info(f"S1 runner completed in {decision.duration} seconds")
    logger.info(f"Validation summary is saved at: {DECISION_OUTPUT_PATH}/{output_dir}")


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
