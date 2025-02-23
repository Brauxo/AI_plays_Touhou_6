from game import TouhouEnv
from model import DQN
from config import EPSILON_START, EPSILON_MIN, EPSILON_DECAY,ACTIONS
import numpy as np

def train():
    env = TouhouEnv()
    agent = DQN()
    epsilon = EPSILON_START
    episodes = 1000

    for episode in range(episodes):
        state = env.capture_screen()
        total_reward = 0
        done = False

        while not done:
            action = agent.get_action(state, epsilon)
            print(f"Chosen Action: {ACTIONS[action]}")
            next_state, reward, done = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.train()
            state = next_state
            total_reward += reward

        agent.update_target()
        if epsilon > EPSILON_MIN:
            epsilon *= EPSILON_DECAY
        print(f"Episode {episode + 1}/{episodes}, Reward: {total_reward}, Epsilon: {epsilon:.3f}")

    agent.save()

if __name__ == "__main__":
    train()