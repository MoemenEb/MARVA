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
from utils.save_runner_csv import save_runner_csv
from entity.decision import Decision


DECISION_OUTPUT_PATH = Path("out/s3_decisions/")
FRAMEWORK = "MARVA v1.0"
LOGGER = "marva.s3.runner"


def main(mode: str, scope: str, limit: int | None):

    setup_logging(run_id="s3_run_" + datetime.now().strftime('%Y%m%d'))
    init_s3_logger()
    logger = logging.getLogger(LOGGER)
    logger.info("Starting S3 runner (mode=%s, scope=%s, limit=%s)", mode, scope, limit)

    # -----------------------------
    # Load dataset
    # -----------------------------
    t0 = time.perf_counter()
    requirement_set = load_dataset(scope, limit)
    logger.info("Loaded %d requirements from '%s' in %.2fs", len(requirement_set.requirements), scope, time.perf_counter() - t0)

    # -----------------------------
    # Init agents + graph
    # -----------------------------
    t0 = time.perf_counter()
    agents = build_agents(mode)
    agents_elapsed = time.perf_counter() - t0
    logger.info("Agents for mode='%s' built in %.2fs (%d agents)", mode, agents_elapsed, len(agents))

    t0 = time.perf_counter()
    graph = build_marva_s3_graph(agents)
    app = graph.compile()
    logger.debug("Graph compiled in %.2fs", time.perf_counter() - t0)

    decision = Decision(
        framework=FRAMEWORK,
        mode=mode
    )

    # -----------------------------
    # Execute
    # -----------------------------
    start_time = time.perf_counter()
    logger.info("Starting S3 pipeline execution")

    if mode == "single":
        total = len(requirement_set.requirements)
        for idx, req in enumerate(requirement_set.requirements, 1):
            req_start = time.perf_counter()
            logger.info("[%d/%d] Processing requirement '%s'", idx, total, req.id)
            state = {
                "mode": "single",
                "requirement": req,
            }
            app.invoke(state)
            req_elapsed = time.perf_counter() - req_start
            req.duration_seconds = round(req_elapsed, 3)
            logger.info("[%d/%d] Requirement '%s' => %s (%.2fs)", idx, total, req.id, req.final_decision, req_elapsed)

    elif mode == "group":
        logger.info("Running group validation for %d requirements", len(requirement_set.requirements))
        state = {
            "mode": "group",
            "requirement_set": requirement_set,
        }
        app.invoke(state)
        logger.info("Group validation => %s", requirement_set.final_decision)

    pipeline_elapsed = time.perf_counter() - start_time
    logger.info("S3 pipeline finished in %.2fs", pipeline_elapsed)

    # -----------------------------
    # Save results
    # -----------------------------
    decision.duration = int(time.perf_counter() - start_time)
    decision.set_decision(requirement_set)

    output_dir = save_runner_decision(decision.to_dict(), DECISION_OUTPUT_PATH)
    csv_path = save_runner_csv(requirement_set, mode, decision.duration, output_dir)
    summary_path = output_dir / "summary.json"
    if mode == "single":
        detailed_path = output_dir / "detailed.json"
        logger.info("S3 runner completed in %ds | output: %s, %s, %s", decision.duration, summary_path, detailed_path, csv_path)
    else:
        logger.info("S3 runner completed in %ds | output: %s, %s", decision.duration, summary_path, csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run S3 (MARVA)")
    parser.add_argument("--scope", required=True)
    parser.add_argument("--mode", required=True, choices=["single", "group"])
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()
    main(args.mode, args.scope, args.limit)
