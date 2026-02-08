# MARVA Codebase Review & Action Plan

## Project Summary

MARVA (Multi-Agent Requirements Validation Architecture) is a Python-based AI system for validating software requirements using LLMs. It implements a multi-stage validation pipeline (S1, S2, S3) with agents that assess requirements across dimensions: atomicity, clarity, completeness, consistency, and redundancy.

---

## Critical Issues

### 1. Shared Mutable Class-Level `llm_response` in `LLMClient`
**File:** `common/llm_client.py:23-30`

`llm_response` is a class variable, not an instance variable. All `LLMClient` instances share the same dict. Since `build_agents()` creates 8 separate instances, concurrent or interleaved calls corrupt each other's response data. The dict is mutated in `generate()` and returned by reference.

**Fix:** Move to `__init__`, return a copy.

### 2. Config Files Exist but Are Never Loaded
**Files:** `config/model.yaml`, `config/global.yaml`, `config/paths.yaml`

All runners hardcode `host="http://localhost:11434"` and `model="qwen3:1.7b"` while `config/model.yaml` specifies `llama3.2`. Config YAML files are dead weight. Settings like `max_retries: 3` and `timeout_seconds: 60` are ignored.

**Fix:** Build a config loader module and wire it into all runners.

### 3. Unsafe Key Access on LLM Responses
**Files:** `s1/pipeline.py:28`, `s2/validation_agents.py:67`, `s3/agents/atomicity_agent.py:20`, `s3/agents/consistency_agent.py:14`, `s3/agents/decision_agent.py:90`

When the LLM times out or errors, `"text"` is `None`. Passing `None` to `extract_json_block` or `.replace()` raises `TypeError`.

**Fix:** Check `execution_status` before accessing `"text"`.

### 4. Silent Error Swallowing in Normalization
**File:** `utils/normalization.py:15-18`

Bare `except Exception: pass` catches all exceptions silently with no logging.

**Fix:** Catch `json.JSONDecodeError` specifically, add `logger.debug`.

---

## High-Severity Issues

### 5. Missing `@staticmethod` on `Reader.get_reader()`
**File:** `utils/reader/reader.py:9`

### 6. Relative Path Fragility
**Files:** `common/prompt_loader.py:3`, `utils/dataset_loader.py:5`

Running from non-root directory causes `FileNotFoundError`.

### 7. Prompt File Name Mismatch
Loads `consistency_single` but file is `consistency_single_arch.txt`.

---

## Medium-Severity Issues

### 8. Inconsistent `AgentResult.to_dict()` Key Casing
**File:** `entity/agent.py:8-12` — Uses `"Agent"`, `"Status"`, `"Issues"` (capitalized).

### 9. Dead Code in `RequirementSet.to_dict()`
**File:** `entity/requirement_set.py:16-18` — Unused `req` dict comprehension.

### 10. Type Mismatch: `RequirementSet.recommendations`
**File:** `entity/requirement_set.py:8` — Initialized as `{}` but comment says `list[str]`.

### 11. `.DS_Store` Committed to Repo

### 12. Unpinned Dependencies in `requirements.txt`

### 13. Inconsistent Logging Levels (DEBUG in S1/S2, INFO in S3)

### 14. Missing `__init__.py` in `s2/`

---

## Low-Severity Issues

### 15. Empty `README.md`
### 16. No `pyproject.toml` or Packaging Setup
### 17. `python-dotenv` in Requirements but Unused
### 18. No Test Infrastructure
### 19. No CI/CD Configuration

---

## Action Plan

| # | Priority | Action | Files Affected |
|---|----------|--------|----------------|
| 1 | **Critical** | Move `llm_response` from class var to instance var, return copy | `common/llm_client.py` |
| 2 | **Critical** | Build config loader, wire into all runners, remove hardcoded values | `common/config.py` (new), `s1/runner.py`, `s2/runner.py`, `s3/agents/__init__.py` |
| 3 | **Critical** | Add defensive checks for LLM error/timeout before accessing `"text"` | `s1/pipeline.py`, `s2/validation_agents.py`, all S3 agents |
| 4 | **Critical** | Replace bare `except Exception` with `json.JSONDecodeError`, add logging | `utils/normalization.py` |
| 5 | **High** | Add `@staticmethod` to `Reader.get_reader()` | `utils/reader/reader.py` |
| 6 | **High** | Use `__file__`-based path resolution for prompts and data | `common/prompt_loader.py`, `utils/dataset_loader.py` |
| 7 | **High** | Fix prompt name mismatch (`consistency_single` vs `consistency_single_arch`) | `prompts/` or `s3/agents/__init__.py` |
| 8 | **Medium** | Lowercase `AgentResult.to_dict()` keys, update consumers | `entity/agent.py`, `s3/agents/decision_agent.py` |
| 9 | **Medium** | Remove dead `req` variable in `RequirementSet.to_dict()` | `entity/requirement_set.py` |
| 10 | **Medium** | Fix `recommendations` init to `[]` | `entity/requirement_set.py` |
| 11 | **Medium** | Add `.DS_Store` to `.gitignore`, remove committed file | `.gitignore` |
| 12 | **Medium** | Pin dependency versions | `requirements.txt` |
| 13 | **Medium** | Standardize logging levels across stages | `s1/logger.py`, `s2/logger.py`, `s3/logger.py` |
| 14 | **Medium** | Add `__init__.py` to `s2/` | `s2/__init__.py` (new) |
| 15 | **Low** | Write proper README | `README.md` |
| 16 | **Low** | Add `pyproject.toml` | `pyproject.toml` (new) |
| 17 | **Low** | Remove unused `python-dotenv` or add `.env` support | `requirements.txt` |
| 18 | **Low** | Add test infrastructure | `tests/` (new) |
| 19 | **Low** | Add CI/CD pipeline | `.github/workflows/` (new) |

---

## Recommended Execution Phases

**Phase 1 — Stability (items 1, 3, 4):** Fix data corruption in LLMClient, add LLM error handling, fix silent error swallowing.

**Phase 2 — Configuration & Paths (items 2, 5, 6, 7):** Centralize config, fix paths, resolve prompt mismatch.

**Phase 3 — Code Quality (items 8-14):** Naming, dead code, types, git hygiene, dependencies, logging.

**Phase 4 — Infrastructure (items 15-19):** Documentation, packaging, tests, CI/CD.
