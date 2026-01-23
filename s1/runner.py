from datetime import datetime
import json
import argparse
import time
import logging
from pathlib import Path

from common.llm_client import LLMClient
# from common.dataset_selector import filter_requirements
from common.normalization import extract_json_block
from s1.pipeline import S1Pipeline
from utils.reader.reader import Reader
from entity.requirement_set import RequirementSet
from common.logging.setup import setup_logging
from s1.logger import init_s1_logger

# DATA_PATH = Path("data/processed/requirements.json")
DECISON_OUTPUT_PATH = Path("out/s1_decisions/")
RAW_DATA_PATH = Path("data/raw/")

def bootstrap_logging():
    setup_logging(run_id="s1_cli_run")
    init_s1_logger()



def main(mode: str, scope: str, limit: int | None):
    bootstrap_logging()
    logger = logging.getLogger("marva.s1.runner")
    logger.info(f"Starting S1 runner with mode={mode}, scope={scope}, limit={limit}")

    OUT_DIR = Path(f"s1/outputs/{mode}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    DATA_PATH = RAW_DATA_PATH / f"{scope}"
    reader = Reader.get_reader(DATA_PATH)
    requirements_set = reader.read(DATA_PATH)
    if limit:
        requirements = requirements_set[:limit]
    requirementSet = RequirementSet(requirements)

    # with open(DATA_PATH, encoding="utf-8") as f:
    #     requirements = json.load(f)
    # requirements = filter_requirements(requirements, scope)

    # if limit:
    #     requirements = requirements[:limit]

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
    decision_out_dir = Path(DECISON_OUTPUT_PATH / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    decision_out_dir.mkdir(parents=True, exist_ok=True)
    startTime = time.perf_counter()
    # -----------------------------
    # SINGLE SCOPE
    # -----------------------------
    if mode == "single":
        
        for req in requirementSet.requirements:
            validation_summary = {
                "requirement_id": req.id,
                "requirement_text": req.text,
            }
            
            result = pipeline.run_single(req)
            json_result = extract_json_block(result["llm_output"])
            by_agent = {
                a["dimension"]: a["status"]
                for a in json_result["agents"]
            }
            json_result.pop("agents", None)
            json_result.pop("agent", None)
            json_result["by_agent"] = by_agent

            validation_summary = {
                **validation_summary,
                **json_result,
            }

            decision_summary["validation_decision"].append(validation_summary)

            out_file = OUT_DIR / f"{req.id}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            print(f"[S1|single] {req.id} done")
    # -----------------------------
    # GROUP SCOPE
    # -----------------------------
    elif mode == "group":
        result = pipeline.run_group(requirementSet.requirements)
        json_result = extract_json_block(result["llm_output"])
        by_agent = {
                a["dimension"]: a["status"]
                for a in json_result["agents"]
            }
        json_result.pop("agents", None)
        json_result.pop("agent", None)
        json_result["by_agent"] = by_agent
        validation_summary = {
            "requirement_id": [r.id for r in requirementSet.requirements],
            "requirement_text": [r.text for r in requirementSet.requirements],
            **json_result,
        }
        decision_summary["validation_decision"].append(validation_summary)

        out_file = OUT_DIR / "s1_group.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print("[S1|group] group analysis done")

    else:
        raise ValueError(f"Invalid scope: {scope}")

    endTime = time.perf_counter()
    flowlatency = int((endTime - startTime))
    decision_summary["flow_latency_seconds"] = flowlatency
    with open(decision_out_dir / "decision_summary.json", "w", encoding="utf-8") as f:
        json.dump(decision_summary, f, indent=2, ensure_ascii=False)
        
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
