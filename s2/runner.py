from datetime import datetime
import argparse
import logging
from pathlib import Path
import time

from common.llm_client import LLMClient
from common.config import load_config
from s2.validation_agents import ValidatorAgent
from utils.dataset_loader import load_dataset
from common.logging.setup import setup_logging
from s2.logger import init_s2_logger
from utils.save_runner_decision import save_runner_decision
from entity.decision import Decision


DECISION_OUTPUT_PATH = Path("out/s2_decisions/")



def main(mode: str, scope: str, limit: int | None):

    setup_logging(run_id="s2_run_"+datetime.now().strftime('%Y%m%d'))
    init_s2_logger()
    logger = logging.getLogger("marva.s2.runner")
    logger.info("Starting S2 runner (mode=%s, scope=%s, limit=%s)", mode, scope, limit)

    # -----------------------------
    # Load dataset
    # -----------------------------
    t0 = time.perf_counter()
    requirement_set = load_dataset(scope, limit)
    logger.info("Loaded %d requirements from '%s' in %.2fs", len(requirement_set.requirements), scope, time.perf_counter() - t0)

    # -----------------------------
    # Init LLM + agent
    # -----------------------------
    t0 = time.perf_counter()
    cfg = load_config()
    llm = LLMClient(
        host=cfg["model"]["host"],
        model=cfg["model"]["model_name"],
        temperature=cfg["model"]["temperature"],
        timeout=cfg["global"]["timeout_seconds"],
        max_retries=cfg["global"]["max_retries"],
    )
    logger.debug("LLM client initialized in %.2fs", time.perf_counter() - t0)

    agents = ValidatorAgent(llm)

    decision = Decision(
        framework= "S2 Validation Agent v1.0",
        mode=mode
    )

    # -----------------------------
    # Execute
    # -----------------------------

    start_time = time.perf_counter()

    logger.info("Starting S2 pipeline execution")
    agents.run(mode=mode, requirement_set=requirement_set)

    pipeline_elapsed = time.perf_counter() - start_time
    decision.duration = int(pipeline_elapsed)
    logger.info("S2 pipeline finished in %.2fs", pipeline_elapsed)

    decision.set_decision(requirement_set)

    output_dir = save_runner_decision(decision.to_dict(), DECISION_OUTPUT_PATH)
    summary_path = output_dir / "summary.json"
    if mode == "single":
        detailed_path = output_dir / "detailed.json"
        logger.info("S2 runner completed in %ds | output: %s, %s", decision.duration, summary_path, detailed_path)
    else:
        logger.info("S2 runner completed in %ds | output: %s", decision.duration, summary_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run S2 baseline")
    parser.add_argument("--scope", required=True)
    parser.add_argument("--mode", required=True, choices=["single", "group"])
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()
    main(args.mode, args.scope, args.limit)
