from collections import defaultdict

def group_requirements(requirements: list) -> dict:
    groups = defaultdict(list)

    for req in requirements:
        groups[req["group_id"]].append(req)

    return dict(groups)
