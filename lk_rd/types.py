from dataclasses import dataclass

import numpy as np


@dataclass
class Prediction:
    mask: np.ndarray
    bbox: np.ndarray | None
    confidence: float
    source: str

