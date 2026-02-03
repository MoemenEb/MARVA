class BaseValidationAgent:

    def __init__(self, llm):
        self.llm = llm
        self.role = self.__class__.__name__

    def run(self, input_data):
        raise NotImplementedError
