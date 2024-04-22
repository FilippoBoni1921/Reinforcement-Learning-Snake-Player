import random
import numpy as np
import tensorflow as tf
import argparse
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

"""
use_gpu = True
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

from a2c_agent import Agent_A2C
from dqn_agent import Agent_DQN,fill_memory
from ppo_agent import Agent_PPO

N = 1000
GAMMA = .9
ITERATIONS = 10000


#algorithm = "ppo"
def training(algorithm,N=N,GAMMA=GAMMA,ITERATIONS=ITERATIONS):

    env_ = get_env(N)

    print("Algortihm:",algorithm)

    rewards_history = [0]
    length_history = [1]
    fruit_history = [0]
    eaten_b_history = [0]
    hit_history = [0]
    win_history = [0]

    if algorithm == "ppo":
        agent = Agent_PPO(len_actions = 4, batch_size=N, discount=GAMMA, actor_rep=5,critic_rep=1)

        for e in range(ITERATIONS):

            if e % 50 == 0:
                print(f"{e}/{ITERATIONS} - {np.mean(rewards_history[-30:]) or 0}, average_length:{np.mean(length_history[-30:])}, fruits eaten:{np.mean(fruit_history[-30:])},bodies eaten boards:{np.mean(eaten_b_history[-30:])},walls hit:{np.mean(hit_history[-30:])}, wins:{np.mean(win_history[-30:])}", end="\n")

            states = env_.to_state()
            
            probs = agent.actor(states)
            
            actions = tf.random.categorical(tf.math.log(probs), 1, dtype=tf.int32)

            rewards,n_fruit,n_body_eaten,n_hit_wall,n_wins = env_.move(actions)

            new_states = env_.to_state()

            agent.learn(states, actions, new_states, rewards)

            if e > 0: 
                rewards_history.append(np.mean(rewards))
                average_length = np.mean([len(sublist)+1 for sublist in env_.bodies])
                length_history.append(average_length)
                fruit_history.append(n_fruit)
                eaten_b_history.append(n_body_eaten)
                hit_history.append(n_hit_wall)
                win_history.append(n_wins)

        agent.save_actor_weights()

        np.save("arrays/reward_history_ppo.npy",rewards_history)
        np.save("arrays/length_history_ppo.npy",length_history)
        np.save("arrays/fruit_history_ppo.npy",fruit_history)
        np.save("arrays/eaten_b_ppo.npy",eaten_b_history)
        np.save("arrays/hit_history_ppo.npy",hit_history)
        np.save("arrays/win_history_ppo.npy",win_history)

        print(f"{e}/{ITERATIONS} - {np.mean(rewards_history[-30:]) or 0}, average_length:{np.mean(length_history[-30:])}, fruits eaten:{np.mean(fruit_history[-30:])},bodies eaten boards:{np.mean(eaten_b_history[-30:])},walls hit:{np.mean(hit_history[-30:])}, wins:{np.mean(win_history[-30:])}", end="\n")


    if algorithm == "dqn":

        batch_size = 2048
        min_replay_size = 5000
        buffer_size = 10000
        epsilon_init = 1.0
        epsilon_min = 0.001
        epsilon_decay = 0.995
        target_update_freq = 10

        agent = Agent_DQN(len_actions=4,epsilon_init=epsilon_init,epsilon_min=epsilon_min,epsilon_decay=epsilon_decay,memory_capacity=buffer_size,discount=GAMMA,batch_size=batch_size)

        fill_memory(env_, agent, min_replay_size,N)

        for e in range(ITERATIONS):
    
            if e % 50 == 0:
                print(f"{e}/{ITERATIONS} - {np.mean(rewards_history[-30:]) or 0}, average_length:{np.mean(length_history[-30:])}, fruits eaten:{np.mean(fruit_history[-30:])},bodies eaten boards:{np.mean(eaten_b_history[-30:])},walls hit:{np.mean(hit_history[-30:])}, wins:{np.mean(win_history[-30:])}", end="\n")

            states = env_.to_state() 

            actions = agent.select_action(states)

            rewards,n_fruit,n_body_eaten,n_hit_wall,n_wins = env_.move(actions)
     
            new_states = env_.to_state()

            agent.memory.store(list(zip(states, actions, rewards, new_states)))

            agent.learn()

            epsilon = agent.update_epsilon()

            if e > 0: 
                rewards_history.append(np.mean(rewards))
                average_length = np.mean([len(sublist)+1 for sublist in env_.bodies])
                length_history.append(average_length)
                fruit_history.append(n_fruit)
                eaten_b_history.append(n_body_eaten)
                hit_history.append(n_hit_wall)
                win_history.append(n_wins)

            if e%target_update_freq == 0:
                agent.update_target()
        
        agent.save_online_weights()
        agent.save_target_weights()

        np.save("arrays/reward_history_dqn.npy",rewards_history)
        np.save("arrays/length_history_dqn.npy",length_history)
        np.save("arrays/fruit_history_dqn.npy",fruit_history)
        np.save("arrays/eaten_b_dqn.npy",eaten_b_history)
        np.save("arrays/hit_history_dqn.npy",hit_history)
        np.save("arrays/win_history_dqn.npy",win_history)

        print(f"{e}/{ITERATIONS} - {np.mean(rewards_history[-30:]) or 0}, average_length:{np.mean(length_history[-30:])}, fruits eaten:{np.mean(fruit_history[-30:])},bodies eaten boards:{np.mean(eaten_b_history[-30:])},walls hit:{np.mean(hit_history[-30:])}, wins:{np.mean(win_history[-30:])}", end="\n")

    if algorithm == "a2c":
        agent = Agent_A2C(len_actions = 4, batch_size=N, discount=GAMMA, actor_rep=1)

        for e in range(ITERATIONS):

            if e % 50 == 0:
                print(f"{e}/{ITERATIONS} - {np.mean(rewards_history[-30:]) or 0}, average_length:{np.mean(length_history[-30:])}, fruits eaten:{np.mean(fruit_history[-30:])},bodies eaten boards:{np.mean(eaten_b_history[-30:])},walls hit:{np.mean(hit_history[-30:])}, wins:{np.mean(win_history[-30:])}", end="\n")

            states = env_.to_state()
            
            probs = agent.actor(states)
            
            actions = tf.random.categorical(tf.math.log(probs), 1, dtype=tf.int32)

            rewards,n_fruit,n_body_eaten,n_hit_wall,n_wins = env_.move(actions)

            new_states = env_.to_state()

            agent.learn(states, actions, new_states, rewards)

            if e > 0: 
                rewards_history.append(np.mean(rewards))
                average_length = np.mean([len(sublist)+1 for sublist in env_.bodies])
                length_history.append(average_length)
                fruit_history.append(n_fruit)
                eaten_b_history.append(n_body_eaten)
                hit_history.append(n_hit_wall)
                win_history.append(n_wins)
        
        agent.save_actor_weights()

        np.save("arrays/reward_history_a2c.npy",rewards_history)
        np.save("arrays/length_history_a2c.npy",length_history)
        np.save("arrays/fruit_history_a2c.npy",fruit_history)
        np.save("arrays/eaten_b_a2c.npy",eaten_b_history)
        np.save("arrays/hit_history_a2c.npy",hit_history)
        np.save("arrays/win_history_a2c.npy",win_history)

        print(f"{e}/{ITERATIONS} - {np.mean(rewards_history[-30:]) or 0}, average_length:{np.mean(length_history[-30:])}, fruits eaten:{np.mean(fruit_history[-30:])},bodies eaten boards:{np.mean(eaten_b_history[-30:])},walls hit:{np.mean(hit_history[-30:])}, wins:{np.mean(win_history[-30:])}", end="\n")


if __name__ == '__main__':
    # if you type --help
    parser = argparse.ArgumentParser(description='Run some functions')

    # Add a command
    parser.add_argument('--training')
    
    # Get our arguments from the user
    args = parser.parse_args()

    if args.training:
        training(args.training)
