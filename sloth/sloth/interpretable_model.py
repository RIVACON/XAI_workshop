from typing import Protocol
import numpy as np

class InterpretableModel(Protocol):
    def predict(self, data:np.ndarray)->np.ndarray:
        pass