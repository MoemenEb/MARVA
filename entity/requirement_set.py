class RequirementSet:
    def __init__(self, requirements: list):
        self.requirements = requirements

        self.group_validations = []

        self.final_decision = None
        self.recommendations = []

    def join_requirements(self) -> str:
        return "\n".join(
            f"[{r.id}] {r.text}" for r in self.requirements
            )
    
    def to_dict(self):
        return {
            "reqs" : self.join_requirements(),
            "status" : self.final_decision,
            "group_validations": self.group_validations,
            "recommendations" : self.recommendations

        }