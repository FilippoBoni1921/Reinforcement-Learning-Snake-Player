
import random
import numpy as np
from  tqdm import trange
import collections
import tensorflow as tf
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

def fill_memory(env,agent,min_replay_size,N):
    
    for i in trange(int(min_replay_size//N)):
        states = env.to_state()
        probs = tf.convert_to_tensor([[.25]*4]*env.n_boards)

        actions =  tf.random.categorical(tf.math.log(probs), 1, dtype=tf.int32)

        rewards,_,_,_,_ = env.move(actions)

        new_states = env.to_state()
        agent.memory.store(list(zip(states, actions, rewards, new_states)))

class Replay_buffer():
    def __init__(self, capacity):
      self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)
    
    def store(self, transitions):
        self.buffer.extend(transitions)
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size,replace=False)
        
        sampled_data = [self.buffer[idx] for idx in indices]
        states, actions, rewards, new_states = zip(*sampled_data)

        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        rewards = tf.convert_to_tensor(rewards)
        new_states = tf.convert_to_tensor(new_states)
                
        return states, actions, rewards, new_states



class Agent_DQN:
    def __init__(self,len_actions,memory_capacity,epsilon_init = 1.0,epsilon_min=0.001,epsilon_decay=0.995,batch_size=256, discount=0.99, step_size=1e-3):
        
        self.memory = Replay_buffer(capacity=memory_capacity)

        self.epsilon = epsilon_init
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.optimizer = tf.optimizers.legacy.Adam(step_size)
        self.batch_size = batch_size
        self.discount = discount
        
        self.len_actions = len_actions

        self.online_net = tf.keras.Sequential( [
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(7, 7, 4), padding='same'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.Dense(64),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.Dense(self.len_actions, activation="linear",
                                  kernel_initializer=tf.initializers.RandomNormal(stddev=0.005),
                                  bias_initializer=tf.initializers.RandomNormal(stddev=0.005))
            ])

        self.target_net = tf.keras.Sequential( [
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(7, 7, 4), padding='same'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.Dense(64),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.Dense(self.len_actions, activation="linear",
                                  kernel_initializer=tf.initializers.RandomNormal(stddev=0.005),
                                  bias_initializer=tf.initializers.RandomNormal(stddev=0.005))
            ])
    def select_action(self,states):

        if not isinstance(states, tf.Tensor):
            states = tf.constant(states, dtype=tf.float32)
            states = tf.convert_to_tensor(states, dtype=tf.float32)

        # Generate random numbers for epsilon-greedy strategy
        random_numbers = tf.random.uniform((states.shape[0],), minval=0, maxval=1)

        # Create a mask for random actions based on epsilon
        random_action_mask = random_numbers <= self.epsilon

        # Generate random actions for the masked states
        random_actions = tf.random.uniform((states.shape[0],), minval=0, maxval=self.len_actions, dtype=tf.int32)

        q_values = self.online_net(states)
        network_actions = tf.cast(tf.argmax(q_values, axis=-1), dtype=tf.int32)

        # Combine random and network actions using tf.where
        actions = tf.where(random_action_mask, random_actions, network_actions)

        return tf.reshape(actions, shape=(-1, 1))

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min,self.epsilon_decay*self.epsilon)
        
        return self.epsilon

    def save_online_weights(self, file_path='dqn_online_weights.h5'):

        self.online_net.save_weights(file_path)
    
    def save_target_weights(self, file_path='dqn_target_weights.h5'):
        # Save the weights of the actor model
        self.target_net.save_weights(file_path)
    

    def learn(self):

        states, actions, rewards, new_states = self.memory.sample(self.batch_size)
        
        q_target = tf.stop_gradient(self.target_net(new_states))
        max_q_target = tf.reduce_max(q_target, axis=1)
            
        # Calculate y_j
        rewards = tf.reshape(rewards, -1)
        y_j = tf.stop_gradient(rewards + (self.discount * max_q_target))
        y_j = tf.reshape(y_j, (-1, 1))

        # Calculate the mean-squared error loss
        with tf.GradientTape() as tape:
            q_pred = self.online_net(states)
            actions_mask =  tf.one_hot(actions[:, 0], depth=4)
            q_pred_actions = tf.reduce_sum(q_pred*actions_mask, axis=-1, keepdims=True)
            loss = tf.losses.mean_squared_error(q_pred_actions,y_j)[:, None]
            loss = tf.reduce_mean(loss)
            
        grad = tape.gradient(loss, self.online_net.trainable_weights)
        self.optimizer.apply_gradients(zip(grad, self.online_net.trainable_weights))
    
    
    def update_target(self):

        # Get the weights from the online net
        online_net_weights = self.online_net.get_weights()

        # Set the weights of the target net
        self.target_net.set_weights(online_net_weights)