import numpy as np

class ComparisonModule:
    def __init__(self, name="ComparisonModule"):
        self.name = name
        
    def calculate_similarity(self, x:np.ndarray, y:np.ndarray) -> float:
        raise NotImplementedError("calculate_similarity not implemented")
    
__all__ = ["ComparisonModule"]