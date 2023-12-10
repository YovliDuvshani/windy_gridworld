from dataclasses import dataclass
from typing import Tuple

import numpy as np

WIND_EFFECT = np.array([0, 0, 0, 1, 1, 1, 2, 2, 1, 0])


@dataclass
class Grid:
    shape: Tuple[int, int]
    start: np.array
    end: np.array


BASE_GRID = Grid(shape=(7, 10), start=np.array([3, 0]), end=np.array([3, 7]))
