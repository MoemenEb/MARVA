
SMALL_DATASETS = {"leeds", "dronology", "reqview", "wasp"}
LARGE_DATASETS = {"qure", "promised"}
ALL_DATASETS = SMALL_DATASETS | LARGE_DATASETS

def filter_requirements(requirements: list, mode: str) -> list:
    """
    Filters requirements by execution mode.

    Supported modes:
      - 'small'   : all small datasets
      - 'large'   : all large datasets
      - 'all'     : all datasets
      - dataset name (e.g., 'wasp', 'reqview', 'qure', ...)
    """
    mode = mode.lower()

    if mode == "all":
        return requirements

    if mode == "small":
        allowed = SMALL_DATASETS

    elif mode == "large":
        allowed = LARGE_DATASETS

    elif mode in ALL_DATASETS:
        allowed = {mode}

    else:
        raise ValueError(
            f"Unknown execution mode '{mode}'. "
            f"Allowed: all | small | large | {', '.join(sorted(ALL_DATASETS))}"
        )

    return [r for r in requirements if r["source"] in allowed]
