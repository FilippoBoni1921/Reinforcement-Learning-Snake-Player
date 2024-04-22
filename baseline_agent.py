import random
import numpy as np
import tensorflow as tf
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

class Agent_baseline:
    def __init__(self):
        
        self.UP = 0
        self.RIGHT = 1
        self.DOWN = 2
        self.LEFT = 3
        
    def choose_actions(self,states,N):
        """
        Proximal Policy Optimization (PPO) implementation using TD(0)
        """
        states_argmax = np.argmax(states,axis = -1)
        actions = np.full(N, 4, dtype=int)

        heads = np.argwhere(states_argmax == 3)
        fruits = np.argwhere(states_argmax == 1)

        dx = fruits[:,1] - heads[:,1]
        dy = fruits[:,2] - heads[:,2]

        for i in range(len(states)):
            if dy[i] > 0 : 
                if dx[i] > 0:
                    possible_actions = np.array([0, 1])
                    chosen_action = np.random.choice(len(possible_actions))
                    actions[i] = possible_actions[chosen_action]

                elif dx[i] < 0:
                    possible_actions = np.array([1, 2])
                    chosen_action = np.random.choice(len(possible_actions))
                    actions[i] = possible_actions[chosen_action]

                else:
                    actions[i] = 1

            elif dy[i] < 0:
                if dx[i] >0:
                    possible_actions = np.array([0, 3])
                    chosen_action = np.random.choice(len(possible_actions))
                    actions[i] = possible_actions[chosen_action]
                elif dx[i] < 0:
                    possible_actions = np.array([2,3])
                    chosen_action = np.random.choice(len(possible_actions))
                    actions[i] = possible_actions[chosen_action]
                
                else:
                    actions[i] = 3
                
            
            elif dy[i] == 0:
                if dx[i] >0:
                    actions[i] = 0
                elif dx[i] <0:
                    actions[i] = 2

        
        return actions