class Predictor:
    def __init__(self, name: str, objective: str):
        self.name = name          # e.g. "dummy" / "linear"
        self.objective = objective

    def suggest(self) -> int:
        """Tell the CLI which batch_size to try next"""
        raise NotImplementedError

    def update(self, batch: int, metrics: dict):
        """The CLI feeds the actual results here so the predictor can learn (can be empty implementation)"""
        pass

    def best(self) -> int:
        """Return the currently estimated optimal batch_size"""
        raise NotImplementedError
