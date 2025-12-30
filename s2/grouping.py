from collections import defaultdict

def group_requirements(requirements: list) -> dict:
    """
    Groups requirements by group_id.

    This is a structural helper, not an agent.
    It prepares context for group-scope validation.

    Returns:
        {
            group_id: [req1, req2, ...],
            None: [ungrouped requirements]
        }
    """
    groups = defaultdict(list)

    for req in requirements:
        groups[req["group_id"]].append(req)

    return dict(groups)
