from agent import Agent
from env import Env

if __name__ == "__main__":
    env = Env()
    agent = Agent(env)
    agent.sarsa_policy_iteration()
    print(1)
