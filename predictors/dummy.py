from .base import Predictor

class DummyPredictor(Predictor):
    def __init__(self, objective: str):
        super().__init__("dummy", objective)
    def suggest(self) -> int:
        return 4
    def update(self, *args, **kwargs):
        pass
    def best(self) -> int:
        return 4
