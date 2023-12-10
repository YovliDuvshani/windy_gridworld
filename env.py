from typing import Tuple, Optional

import numpy as np

from config import POSSIBLE_ACTIONS_MAPPING
from grid import WIND_EFFECT, Grid, BASE_GRID
from utils import REWARD, TERMINAL


class Env:
    def __init__(
        self,
        wind_effect: Optional[np.array] = WIND_EFFECT,
        grid: Optional[Grid] = BASE_GRID,
    ):
        self.wind_effect = wind_effect
        self.grid = grid

    def initialize_new_episode(self):
        return self.grid.start

    def transitions(
        self, state: np.array, action: np.array
    ) -> Tuple[np.array, REWARD, TERMINAL]:
        next_state = state + action
        next_state[0] -= self.wind_effect[state[1]]
        next_state[0] = np.clip(next_state[0], 0, self.grid.shape[0] - 1)
        next_state[1] = np.clip(next_state[1], 0, self.grid.shape[1] - 1)

        return next_state, -1, self._is_terminal_state(next_state)

    def _is_terminal_state(self, state: np.array):
        if np.array_equal(state, self.grid.end):
            return True
        return False

    @staticmethod
    def possible_actions_mapping():
        return POSSIBLE_ACTIONS_MAPPING
