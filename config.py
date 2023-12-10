import numpy as np

NUMBER_OF_EPISODES = 5000
ALPHA = 0.1
EPSILON_GREEDY_RATE = 0.9

POSSIBLE_ACTIONS_MAPPING = {
    0: np.array([1, 0]),
    1: np.array([0, 1]),
    2: np.array([-1, 0]),
    3: np.array([0, -1]),
    # 4: np.array([1, -1]),
    # 5: np.array([-1, 1]),
    # 6: np.array([1, 1]),
    # 7: np.array([-1, -1]),
    # 8: np.array([0, 0]),
}

USE_STOCHASTIC_WIND = True
