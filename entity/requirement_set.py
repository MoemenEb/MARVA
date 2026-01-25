class RequirementSet:
    def __init__(self, requirements: list):
        self.requirements = requirements

        self.pairwise_validations = {}
        self.group_validations = {}

        self.final_decision = None
        self.recommendations = []   # list[str]

    def join_requirements(self) -> str:
        return "\n".join(
            f"[{r.id}] {r.text}" for r in self.requirements
            )