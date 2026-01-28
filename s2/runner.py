from datetime import datetime
import argparse
import logging
from pathlib import Path
import time

from common.llm_client import LLMClient
from s2.validation_agents import ValidatorAgent
from utils.dataset_loader import load_dataset
from common.logging.setup import setup_logging
from s2.logger import init_s2_logger
from utils.save_runner_decision import save_runner_decision
from entity.decision import Decision


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

    agents = ValidatorAgent(llm)
    
    decision = Decision(
        framework= "S2 Validation Agent v1.0",
        mode=mode
    )

    # -----------------------------
    # Execute
    # -----------------------------

    startTime = time.perf_counter()
    
    logger.info("Start S2 pipeline")
    agents.run(mode=mode, requirement_set=requirement_set)

    decision.duration = int((time.perf_counter() - startTime))
    logger.info("S2 pipeline finished")
    
    logger.info("Saving results ...")
    decision.set_decision(requirement_set)
    
    dir = save_runner_decision(decision.to_dict(), DECISON_OUTPUT_PATH)

    logger.info(f"S2 runner completed in {decision.duration} seconds")
    logger.info(f"Validation summary is saved at: {DECISON_OUTPUT_PATH}/{dir}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run S2 baseline")
    parser.add_argument("--scope", required=True)
    parser.add_argument("--mode", required=True, choices=["single", "group"])
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()
    main(args.mode, args.scope, args.limit)
