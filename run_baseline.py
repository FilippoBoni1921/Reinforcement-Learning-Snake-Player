
import random
import numpy as np
import tensorflow as tf
import argparse
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)


import environment_fully_observable 
import environment_partially_observable

from environment_definition_for_training import get_env

from a2c_agent import Agent_A2C
from dqn_agent import Agent_DQN,fill_memory
from ppo_agent import Agent_PPO
from baseline_agent import Agent_baseline

"""
use_gpu = False
if use_gpu == False:
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        # Disable all GPUS
        tf.config.set_visible_devices([], 'GPU')
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != 'GPU'
    except:
    # Invalid device or cannot modify virtual devices once initialized.
        pass

gpu = len(tf.config.get_visible_devices('GPU'))>0
print("GPU is", "available" if gpu else "NOT AVAILABLE")
"""


def run_baseline(N,ITERATIONS,run="eval"):

    env_ = get_env(N)

    print("Algortihm: Baseline")

    rewards_history = [0]
    length_history = [1]
    fruit_history = [0]
    eaten_b_history = [0]
    hit_history = [0]
    win_history = [0]

    agent = Agent_baseline()

    for e in range(ITERATIONS):
        if e % 50 == 0:
            if run == "eval":
                print(f"{e}/{ITERATIONS} - {np.mean(rewards_history[-30:]) or 0}, average_length:{np.mean(length_history[-30:])}", end="\n")
            elif run == "training":
                print(f"{e}/{ITERATIONS} - {np.mean(rewards_history[-30:]) or 0}, average_length:{np.mean(length_history[-30:])}, fruits eaten:{np.mean(fruit_history[-30:])},bodies eaten boards:{np.mean(eaten_b_history[-30:])},walls hit:{np.mean(hit_history[-30:])}, wins:{np.mean(win_history[-30:])}", end="\n")

        states = env_.to_state()
    
        actions = agent.choose_actions(states,N)
        # act on each env and collect reward/done
        rewards,n_fruit,n_body_eaten,n_hit_wall,n_wins = env_.move(actions.reshape(-1,1))

        if e > 0:
            if run == "eval":
                rewards_history.append(np.mean(rewards))
                average_length = np.mean([len(sublist)+1 for sublist in env_.bodies])
                length_history.append(average_length)
            elif run == "training":
                rewards_history.append(np.mean(rewards))
                average_length = np.mean([len(sublist)+1 for sublist in env_.bodies])
                length_history.append(average_length)
                fruit_history.append(n_fruit)
                eaten_b_history.append(n_body_eaten)
                hit_history.append(n_hit_wall)
                win_history.append(n_wins)
    
    if run == "eval":
        mean_rewards_history = np.mean(rewards_history)
        mean_length_history = np.mean(length_history)
        np.save("mean_reward_baseline.npy",mean_rewards_history)
        np.save("mean_length_baseline.npy",mean_length_history)
        
        print(f"{e}/{ITERATIONS} - {np.mean(rewards_history[-30:]) or 0}, average_length:{np.mean(length_history[-30:])}", end="\n")
            
    elif run == "training":
        mean_rewards_history = np.mean(rewards_history)
        mean_length_history = np.mean(length_history)
        mean_fruit_history = np.mean(fruit_history)
        mean_eaten_b_history = np.mean(eaten_b_history)
        mean_hit_history = np.mean(hit_history)
        mean_win_history = np.mean(win_history)

        np.save("arrays/mean_reward_baseline.npy",mean_rewards_history)
        np.save("arrays/mean_length_baseline.npy",mean_length_history)
        np.save("arrays/mean_fruit_baseline.npy",mean_fruit_history)
        np.save("arrays/mean_eaten_b_baseline.npy",mean_eaten_b_history)
        np.save("arrays/mean_hit_history_baseline.npy",mean_hit_history)
        np.save("arrays/mean_win_history_baseline.npy",mean_win_history )

        print(f"{e}/{ITERATIONS} - {np.mean(rewards_history[-30:]) or 0}, average_length:{np.mean(length_history[-30:])}, fruits eaten:{np.mean(fruit_history[-30:])},bodies eaten boards:{np.mean(eaten_b_history[-30:])},walls hit:{np.mean(hit_history[-30:])}, wins:{np.mean(win_history[-30:])}", end="\n")

if __name__ == '__main__':
    # if you type --help
    parser = argparse.ArgumentParser(description='Run some functions')

    # Add a command
    parser.add_argument('--run_baseline', nargs='+', metavar=('arguments'))

    # Get our arguments from the user
    args = parser.parse_args()

    if args.run_baseline:
        parsed_args = dict(item.split('=') for item in args.run_baseline)

        N = int(parsed_args.get('N', 0))
        ITERATIONS = int(parsed_args.get('ITERATIONS', 0))
        run = parsed_args.get('run', 'eval')

        run_baseline(N, ITERATIONS, run)

    


