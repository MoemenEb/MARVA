from datetime import datetime
import json
import argparse
import logging
from pathlib import Path

from common.llm_client import LLMClient
from common.logging.setup import setup_logging
from s2.validation_agents import ValidatorAgent
from s2.validation_summary import ValidatorSummary
from utils.reader.reader import Reader
from entity.requirement_set import RequirementSet
from s2.logger import init_s2_logger

DATA_PATH = Path("data/processed/requirements_grouped.json")
DECISION_OUTPUT_PATH = Path("out/s2_decisions/")
RAW_DATA_PATH = Path("data/raw/")


def calculate_total_latency(results: dict) -> int:
    """Calculate total latency from validation results."""
    return sum(item.get('latency_ms', 0) for item in results.values() if isinstance(item, dict))


def process_validation(validator: ValidatorAgent, summarizer: ValidatorSummary, 
                       mode: str, item):
    """Execute validation and summary for single requirement or group."""
    is_single = mode == "single"
    
    
    # Execute validation
    validation_results = validator.execute(
        mode=mode,
        requirement=item.requirements if is_single else None,
        group=item if not is_single else None
    )
    summary = summarizer.summarize(validation_results, item)
    
    # Extract item-specific info
    if is_single:
        item_id = item["req_id"]
        base_info = {
            "req_id": item["req_id"],
            "requirement_text": item["text"],
            "source": item["source"],
        }
        decision_info = {
            "requirement_id": item["req_id"],
            "requirement_text": item["text"],
        }
    else:
        item_id = item[0]["group_id"]
        base_info = {
            "group_id": item_id,
            "source": item[0]["source"],
            "requirement_ids": [r["req_id"] for r in item],
            "requirement_texts": [r["text"] for r in item],
        }
        decision_info = {
            "requirements": [
                {"requirement_id": r.get("req_id", "unknown"), 
                 "requirement_text": r.get("text", "unknown")}
                for r in item
            ]
        }
    
    # Calculate metrics
    validation_latency = calculate_total_latency(validation_results)
    flow_latency = (validation_latency + summary["latency_ms"]) / 1000
    
    common_metrics = {
        "validation_latency_ms": validation_latency,
        "summary_latency_ms": summary["latency_ms"],
        "flow_latency_seconds": flow_latency
    }
    
    # Build result and decision
    result = {
        "mode": mode,
        **base_info,
        "results": validation_results,
        "summary": summary,
        **common_metrics
    }
    
    decision = {
        **decision_info,
        "decision": summary["output"]["final_status"],
        "by_agent": {k: v["output"]["decision"] for k, v in validation_results.items()},
        "recommendations": summary["output"]["recommendations"],
        **common_metrics
    }
    
    return result, decision, item_id, flow_latency


def save_json(path: Path, data: dict):
    """Save data as JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main(mode: str, scope: str, limit: int | None):
    # Load and filter dataset
    # with open(DATA_PATH, encoding="utf-8") as f:
    #     requirements = json.load(f)
    
    # requirements = filter_requirements(requirements, scope)[:limit] if limit else filter_requirements(requirements, scope)
    setup_logging(run_id="s2_run_"+datetime.now().strftime('%Y%m%d'))
    init_s2_logger()
    logger = logging.getLogger("marva.s2.runner")

    DATA_PATH = RAW_DATA_PATH / f"{scope}"
    reader = Reader.get_reader(DATA_PATH)
    requirements_set = reader.read(DATA_PATH)
    if limit:
        requirements = requirements_set[:limit]
    requirementSet = RequirementSet(requirements)
    logger.info(f"Starting S2 runner with mode={mode}, scope={scope}, limit={limit}")
    # print(f"[S2] mode={mode}, scope={scope}, count={len(requirements)}")

    # Setup
    # groups = group_requirements(requirements)
    llm = LLMClient(host="http://localhost:11434", model="qwen3:1.7b")
    validator = ValidatorAgent(llm)
    summarizer = ValidatorSummary(llm)

    # Create output directories
    out_dir = Path(f"s2/outputs/{mode}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    decision_out_dir = DECISION_OUTPUT_PATH / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    decision_out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize final decision structure
    final_decision = {
        "mode": mode,
        "scope": scope,
        "Validation framework": "S2 Validation Agent v1.0",
        "total_flow_latency_seconds": 0,
        "Validation Decisions": [],
    }

    # Process items
    # items = requirementSet if mode == "group" else requirementSet.requirements
    total_latency = 0
    
    result, decision, item_id, flow_latency = process_validation(
            validator, summarizer, mode, requirementSet
        )
        
    total_latency += flow_latency
    final_decision["Validation Decisions"].append(decision)
        
    # Save individual result
    filename = f"group_{item_id}.json" if mode == "group" else f"{item_id}.json"
    save_json(out_dir / filename, result)
    
    print(f"[S2-{mode}] {item_id} done")
    
    # Save final decision summary
    final_decision["total_flow_latency_seconds"] = total_latency
    save_json(decision_out_dir / "decision_summary.json", final_decision)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run S2 baseline")
    parser.add_argument("--scope", required=True)
    parser.add_argument("--mode", required=True, choices=["single", "group"])
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()
    main(args.mode, args.scope, args.limit)