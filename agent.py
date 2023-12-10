from random import random

import numpy as np

from config import NUMBER_OF_EPISODES, EPSILON_GREEDY_RATE, ALPHA
from env import Env


class Agent:
    def __init__(self, env: Env):
        self.env = env
        self.q = np.zeros(
            (*self.env.grid.shape, len(self.env.possible_actions_mapping()))
        )

    def sarsa_policy_iteration(self):
        for i in range(NUMBER_OF_EPISODES):
            state = self.env.initialize_new_episode()
            steps = []
            terminal = False
            while not terminal:
                steps += [state]
                possible_actions_mapping = self.env.possible_actions_mapping()
                action = self._select_action_based_on_epsilon_greedy_policy(
                    state, possible_actions_mapping
                )
                next_state, reward, terminal = self.env.transitions(
                    state, possible_actions_mapping[action]
                )
                next_action = self._select_action_based_on_epsilon_greedy_policy(
                    next_state, possible_actions_mapping
                )
                self.q[state[0], state[1], action] += ALPHA * (
                    self.q[next_state[0], next_state[1], next_action]
                    + reward
                    - self.q[state[0], state[1], action]
                )
                state, action = next_state, next_action
            print(f"Episode {i} ends with {len(steps)} steps using {steps}")

    def _select_action_based_on_epsilon_greedy_policy(
        self, state: np.array, possible_actions: dict[int, np.array]
    ):
        if random() < 1 - EPSILON_GREEDY_RATE:
            idx_action = np.random.choice(list(possible_actions.keys()))
            return idx_action
        return np.argmax(self.q[state[0], state[1], :])
