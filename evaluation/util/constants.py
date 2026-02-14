from pathlib import Path

# --- Evaluator constants ---
EXCLUDE_COLUMNS = {"id", "requirement", "recommendations", "duration", "notes", "NOTES"}
POS_LABEL = "FAIL"
LABELS = ["PASS", "FAIL"]

# --- Score metrics tracked ---
SCORE_METRICS = ["accuracy", "precision", "recall", "f1"]

# --- Default directories ---
DEFAULT_FIGURES_DIR = Path("figures")
DEFAULT_OUT_DIR = Path("results")

# --- Ground truth mapping ---
GROUND_TRUTH_DIR = Path("data/syntatic")
GROUND_TRUTH_MAP = {
    "single": GROUND_TRUTH_DIR / "single_2.csv",
}

EVAL_MODES = ["scores", "confusion", "both"]
