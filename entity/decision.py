
class Decision:
    def __init__(self, framework:str, mode:str):
        self.framework = framework
        self.mode = mode
        self.duration = 0
        self.decision = {}

    def to_dict(self):
        return{
            "Framework" : self.framework,
            "Mode" : self.mode,
            "Duration" : self.duration,
            "Validation" : self.decision
        }