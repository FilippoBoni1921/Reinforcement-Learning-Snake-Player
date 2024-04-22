import random
import numpy as np

import tensorflow as tf
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

class Agent_A2C:
    def __init__(self,len_actions,batch_size=256, discount=0.99, clip_eps=0.2, step_size=1e-3,
                 actor_rep=15, critic_rep=1,n_nr=64,show_summary = False):
        
        self.len_actions = len_actions
        self.discount = discount
        self.clip_eps = clip_eps
        self.actor_rep = actor_rep
        self.critic_rep = critic_rep

        self.optimizer_actor = tf.optimizers.legacy.Adam(step_size)
        self.optimizer_critic = tf.optimizers.legacy.Adam(step_size)
        self.batch_size = batch_size


        self.actor = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(7, 7, 4), padding='same'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.Dense(64),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.Dense(self.len_actions, activation=tf.nn.softmax,
                                  kernel_initializer=tf.initializers.RandomNormal(stddev=0.005),
                                  bias_initializer=tf.initializers.RandomNormal(stddev=0.005))
        ])
        self.critic = tf.keras.Sequential( [
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(7, 7, 4), padding='same'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.Dense(64),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.Dense(1, activation="linear")
        ])

        if show_summary == True:
            print(self.actor.summary())
            print(self.critic_net.summary())
    
    def save_actor_weights(self, file_path='a2c_actor_weights.h5'):
        # Save the weights of the actor model
        self.actor.save_weights(file_path)
        
    def learn(self, states, actions, new_states, rewards):
        """
        Proximal Policy Optimization (PPO) implementation using TD(0)
        """
        rewards = np.reshape(rewards, (-1, 1))
        actions = tf.one_hot(actions[:, 0], depth=4).numpy()
        val = self.critic(states)
        new_val = self.critic(new_states)
        reward_to_go = tf.stop_gradient(rewards + self.discount * new_val)# * (1-dones))
        td_error = (reward_to_go - val).numpy()

        for _ in range(self.actor_rep):
            indexes = np.random.choice(range(0, len(states)), min(self.batch_size, len(states)), replace=False)
            with tf.GradientTape() as a_tape:
                probs = self.actor(states[indexes])
                selected_actions_probs = tf.reduce_sum(probs * actions[indexes], axis=-1, keepdims=True)

                loss_actor = td_error[indexes] * tf.math.log(selected_actions_probs)
                loss_actor = tf.reduce_mean(-loss_actor)

            grad_actor = a_tape.gradient(loss_actor, self.actor.trainable_weights)
            self.optimizer_actor.apply_gradients(zip(grad_actor, self.actor.trainable_weights))

        for _ in range(self.critic_rep):
            indexes = np.random.choice(range(0, len(states)), min(self.batch_size, len(states)), replace=False)
            with tf.GradientTape() as c_tape:
                val = self.critic(states[indexes])
                new_val = tf.stop_gradient(self.critic(new_states[indexes]))
                reward_to_go = tf.stop_gradient(rewards[indexes] + self.discount * new_val)# * (1-dones))
                loss_critic = tf.losses.mean_squared_error(val, reward_to_go)[:, None]
                loss_critic = tf.reduce_mean(loss_critic)
            grad_critic = c_tape.gradient(loss_critic, self.critic.trainable_weights)
            self.optimizer_critic.apply_gradients(zip(grad_critic, self.critic.trainable_weights))
        

        #return loss_actor,loss_critic
