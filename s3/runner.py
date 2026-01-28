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


def main(mode: str, scope: str, limit: int | None):

    setup_logging(run_id="s3_run_"+datetime.now().strftime('%Y%m%d'))
    init_s3_logger()
    logger = logging.getLogger("marva.s3.runner")
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


    decision_summary = {
        "mode": mode,
        "scope": scope,
        "validation framework" : "MARVA v1.0",
        "flow_latency_seconds": None
    }

    decision = Decision(
        framework= "MARVA v1.0",
        mode= mode
    )

    # -----------------------------
    # Execute (SEMANTIC PARITY WITH S2)
    # -----------------------------
    if mode == "single":
        startTime = time.perf_counter()
        single_out = {
            "Validation Decisions": []
        }
        for req in requirement_set.requirements:
            state = {
                "mode": "single",
                "requirement": req,
            }
            result = app.invoke(state)
            result.pop("requirement")
            decision = result["decision"]

            full = {
                "requirement_id": req.id,
                "requirement_text": req.text,
                **decision
            }
            single_out["Validation Decisions"].append(full)
            logger.info(f"[S3-single] {req.id} done")

        endTime = time.perf_counter()
        flowlatency = int((endTime - startTime))   
        decision_summary["flow_latency_seconds"] = flowlatency 
        final_decision = {
            **decision_summary,
            **single_out,
        }
        save_runner_decision(final_decision, DECISON_OUTPUT_PATH)

    elif mode == "group":
        reqi = {"requirements": []}
        startTime = time.perf_counter()
        state = {
                "mode": "group",
                "group": requirement_set.requirements,
            }
        for r in requirement_set.requirements:
            requir = {
                    'req_id': r.id,
                    'text': r.text
                }
            reqi["requirements"].append(requir)
                

        result = app.invoke(state)
        result.pop("group")
        decision = result["decision"]

        full = {
            "scope": scope,
            **reqi,
            **decision
        }
        endTime = time.perf_counter()
        flowlatency = int((endTime - startTime))
        decision_summary["flow_latency_seconds"] = flowlatency
        final_decision = {
        **decision_summary,
        **full,
        }
        save_runner_decision(final_decision, DECISON_OUTPUT_PATH)
        logger.info(f"[S3-group] done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run S3 (MARVA)")
    parser.add_argument("--scope", required=True)
    parser.add_argument("--mode", required=True, choices=["single", "group"])
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()
    main(args.mode, args.scope, args.limit)