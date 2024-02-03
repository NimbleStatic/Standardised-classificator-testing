import numpy as np
from typing import Callable, List, Dict, Tuple
from numpy import array
from optimisers import BaseOptimiser



class BaseClassificator:
    def __init__(self) -> None:
        pass
    
    def classify_sample(self, x:array):
        pass
    
    def classify_sample_list(self, X:List[array])-> List[float]:
        pass
    
    def train_on_data(self, X:List[array], Y:List[float], optimiser:BaseOptimiser):
        