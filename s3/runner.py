from datetime import datetime
import argparse
import logging
from pathlib import Path
import time

from common.logging.setup import setup_logging
from s3.graph import build_marva_s3_graph
from s3.agents import build_agents
from s3.logger import init_s3_logger
from utils.dataset_loader import load_dataset
from utils.save_runner_decision import save_runner_decision
from entity.decision import Decision


DECISON_OUTPUT_PATH = Path("out/s3_decisions/")
FRAMEWORK = "MARVA v1.0"
LOGGER = "marva.s3.runner"


def main(mode: str, scope: str, limit: int | None):

    setup_logging(run_id="s3_run_" + datetime.now().strftime('%Y%m%d'))
    init_s3_logger()
    logger = logging.getLogger(LOGGER)
    logger.info(f"Starting S3 runner with mode={mode}, scope={scope}, limit={limit}")

    # -----------------------------
    # Load dataset
    # -----------------------------
    logger.info(f"Loading dataset {scope}")
    requirement_set = load_dataset(scope, limit)
    logger.info(f"Loaded {len(requirement_set.requirements)} requirements from dataset")

    # -----------------------------
    # Init agents + graph
    # -----------------------------
    agents = build_agents()
    graph = build_marva_s3_graph(agents)
    app = graph.compile()

    decision = Decision(
        framework=FRAMEWORK,
        mode=mode
    )

    # -----------------------------
    # Execute
    # -----------------------------
    start_time = time.perf_counter()
    logger.info("Start S3 pipeline")

    if mode == "single":
        for req in requirement_set.requirements:
            state = {
                "mode": "single",
                "requirement": req,
            }
            app.invoke(state)
            logger.info(f"[S3-single] {req.id} done")

    elif mode == "group":
        state = {
            "mode": "group",
            "requirement_set": requirement_set,
        }
        app.invoke(state)
        logger.info("[S3-group] done")

    logger.info("S3 pipeline finished")

    # -----------------------------
    # Save results
    # -----------------------------
    logger.info("Saving results ...")
    decision.duration = int(time.perf_counter() - start_time)
    decision.set_decision(requirement_set)

    output_dir = save_runner_decision(decision.to_dict(), DECISON_OUTPUT_PATH)
    logger.info(f"S3 runner completed in {decision.duration} seconds")
    logger.info(f"Validation summary is saved at: {DECISON_OUTPUT_PATH}/{output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run S3 (MARVA)")
    parser.add_argument("--scope", required=True)
    parser.add_argument("--mode", required=True, choices=["single", "group"])
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()
    main(args.mode, args.scope, args.limit)