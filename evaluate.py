import random
import numpy as np
import tensorflow as tf
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

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

import environment_fully_observable 
import environment_partially_observable
from environment_definition_for_training import get_env
from ppo_agent import Agent_PPO
from run_baseline import run_baseline

N = 100
GAMMA = .9
ITERATIONS = 1000

env_ = get_env(N)

### FINAL AGENT ###
rewards_history = [0]
length_history = [1]
win_history = [0]

agent = Agent_PPO(len_actions = 4)
agent.actor.load_weights("ppo_actor_weights.h5")

for e in range(ITERATIONS):

        if e % 50 == 0:
            print(f"{e}/{ITERATIONS} - {np.mean(rewards_history[-30:]) or 0}, average_length:{np.mean(length_history[-30:])}, wins:{np.mean(win_history[-30:])}", end="\n")

        states = env_.to_state()
        
        probs = agent.actor(states)
        
        actions = tf.random.categorical(tf.math.log(probs), 1, dtype=tf.int32)

        rewards,n_fruit,n_body_eaten,n_hit_wall,n_wins = env_.move(actions)

        #if e > 50: 
        if e > 0:
            rewards_history.append(np.mean(rewards))
            average_length = np.mean([len(sublist)+1 for sublist in env_.bodies])
            length_history.append(average_length)
            win_history.append(n_wins)


print(f"{e}/{ITERATIONS} - {np.mean(rewards_history[-30:]) or 0}, average_length:{np.mean(length_history[-30:])}, wins:{np.mean(win_history[-30:])}", end="\n")


### BASELINE ###
run_baseline(N=N,ITERATIONS=ITERATIONS)

### PLOT ###
from plots import plots

title = "Reward Final Comparison"
x_label = ""
y_label = "Reward"
file_name = "final_reward_comparison"

agent = rewards_history
baseline = np.load("mean_reward_baseline.npy")

# Sample data dictionary with Baseline as a single value
data_dict = {
    'PPO': agent,
    'Baseline': baseline  # Example of Baseline as a single value
}

plots(title, x_label, y_label, file_name, data_dict,window_size=10,eval=True,str=50)

title = "Average Length Final Comparison"
x_label = ""
y_label = "Length"
file_name = "final_length_comparison"

agent = length_history
baseline = np.load("mean_length_baseline.npy")


# Sample data dictionary with Baseline as a single value
data_dict = {
    'PPO': agent,
    'Baseline': baseline  # Example of Baseline as a single value
}

plots(title, x_label, y_label, file_name, data_dict,window_size=10,eval=True,str=50)
