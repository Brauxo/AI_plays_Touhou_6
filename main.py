from game import TouhouEnv
from model import DQN
from config import EPSILON_START, EPSILON_MIN, EPSILON_DECAY, ACTIONS
import numpy as np
import time

def train():
    env = TouhouEnv()
    agent = DQN()
    epsilon = EPSILON_START
    episodes = 1000
    
    TRAIN_EVERY_N_STEPS = 4
    UPDATE_TARGET_EVERY_N_EPISODES = 5
    step_count = 0

    try:
        state = env.reset()

        for episode in range(1, episodes + 1):
            total_reward = 0
            episode_done = False
            local_step = 0
            
            while not episode_done:
                step_count += 1
                local_step += 1
                
                action_index = agent.get_action(state, epsilon)
                action_name = ACTIONS[action_index]
                print(f"Ep {episode} | Step {local_step} | Action: {action_name}", end='\r')
                next_state, reward, done = env.step(action_index)
                
                agent.store_transition(state, action_index, reward, next_state, done)
                
                if step_count % TRAIN_EVERY_N_STEPS == 0:
                    agent.train()

                state = next_state
                total_reward += reward

                if done:
                    episode_done = True

            if episode % UPDATE_TARGET_EVERY_N_EPISODES == 0:
                agent.update_target()

            if epsilon > EPSILON_MIN:
                epsilon *= EPSILON_DECAY

            print(f"\nEpisode {episode} Finished | Total Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")

            if episode % 25 == 0:
                print(f"--- Saving model at episode {episode} ---")
                agent.save()

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        print("Cleaning up resources and saving final model...")
        env.cleanup()
        agent.save()

if __name__ == "__main__":
    train()