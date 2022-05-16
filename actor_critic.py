import collections
import gym
import numpy as np
import statistics
import tensorflow as tf
import tqdm
from tensorflow import keras

from matplotlib import pyplot as plt
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple

# Create the environment
env = gym.make("CartPole-v1")

# Small epsilon value for stabilizing division operations used in standardization
eps = np.finfo(np.float32).eps.item()

#Build the architecture, we will assume that the policy head will output the logits of the action probabilites

def build_model(learning_rate = 0.01):
    inputs = keras.Input(shape=(4,))
    x = layers.Dense(128, activation = 'relu')(inputs)   
    outputs_policy = layers.Dense(2, activation = 'linear')(x) 
    outputs_value = layers.Dense(1, activation = 'linear')(x)
    network = keras.Model(inputs=inputs, outputs=[outputs_policy, outputs_value])
    return network

#For all operations to work we need to change some datatypes

def env_step(action):
    state, reward, done, _ = env.step(action)
    return (state.astype(np.float32), np.array(reward, np.int32), np.array(done, np.int32))


def tf_env_step(action):
    return tf.numpy_function(env_step, [action], [tf.float32, tf.int32, tf.int32])

#Define function that runs an episode outputting the results

def run_episode(initial_state, model, max_steps):
        
    states = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
    action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

    initial_state_shape = initial_state.shape
    state = initial_state
    # Convert state into a batched tensor (batch size = 1) to feed it to the network
    state = tf.expand_dims(state, 0)
    
    states.write(0, state)

    for t in tf.range(max_steps):
        # Run the model and to get action probabilities and critic value
        action_logits_t, value = model(state)

        # Sample next action from the action probability distribution
        action = tf.random.categorical(action_logits_t, 1)[0, 0]
        action_probs_t = tf.nn.softmax(action_logits_t)

        # Store critic values
        values = values.write(t, tf.squeeze(value))

        # Store log probability of the action chosen
        action_probs = action_probs.write(t, action_probs_t[0, action])

        # Apply action to the environment to get next state and reward
        state, reward, done = tf_env_step(action)
        state.set_shape(initial_state_shape)
        # Convert state into a batched tensor (batch size = 1) to feed it to the network
        state = tf.expand_dims(state, 0)
        
        states = states.write(t + 1, state)

        # Store reward
        rewards = rewards.write(t, reward)

        if tf.cast(done, tf.bool):
            break

    action_probs = action_probs.stack()
    values = values.stack()
    rewards = rewards.stack()
    states = states.stack()
    
    return action_probs, values, rewards, states

def get_q_estimates(rewards, model, gamma, N, states, standardize = False):
    n = tf.shape(rewards)[0]
    q_estimates = tf.TensorArray(dtype=tf.float32, size=n)

    # Start from the end of `rewards` and accumulate estimate the Q-values
    rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    discounted_sum = tf.constant(0.0)
    discounted_sum_shape = discounted_sum.shape
    for i in tf.range(n):
        T = tf.math.minimum(N, i)
        reward = rewards[i]
        bootstrapped_value = model(states[n-i-1 + T])[1]
        if i< N :
            discounted_sum = reward + gamma * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
        q_estimates = q_estimates.write(i, discounted_sum + bootstrapped_value)
        
    q_estimates = q_estimates.stack()[::-1]

    if standardize:
        q_estimates = ((q_estimates - tf.math.reduce_mean(q_estimates)) / (tf.math.reduce_std(q_estimates) + eps))

    return q_estimates

def compute_loss(action_probs: tf.Tensor,  values: tf.Tensor, q_estimates : tf.Tensor):
    #advantage = returns - values
    action_log_probs = tf.math.log(action_probs)
    actor_loss = -tf.math.reduce_sum(action_log_probs * q_estimates)
    critic_loss = tf.math.reduce_sum((q_estimates - values)**2)

    return actor_loss + critic_loss

@tf.function
def train_step(initial_state, model, optimizer, gamma, max_steps_per_episode, N):
                    
    # Run the model for one episode to collect training data
    action_probs, values, rewards, states = run_episode(initial_state, model, max_steps_per_episode) 

    # Calculate expected returns
    q_values = get_q_estimates(rewards, model, gamma, N, states, standardize = False)

    # Convert training data to appropriate TF tensor shapes
    action_probs, values, q_values = [tf.expand_dims(x, 1) for x in [action_probs, values, q_values]] 

    # Calculating loss values to update our network
    loss = compute_loss(action_probs, values, q_values)
            
    # Calculate episode reward
    episode_reward = tf.math.reduce_sum(rewards)

        # Compute the gradients from the loss
        #grads = tape.gradient(loss, model.trainable_variables)

    # Apply the gradients to the model's parameters
    #optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return episode_reward, loss

# Choose learning rate
learning_rate = 0.001
    
# Bootstrap after steps
N = 36
    
# Choose size of batches of gradients
batch_update_size = 16
    
# Discount factor for future rewards
gamma = 0.99

def cartpole(learning_rate, N, batch_update_size, gamma):
    
    min_episodes_criterion = 100
    max_episodes = 10000
    max_steps_per_episode = 2000 # This is possibly useless, cartpole's standard is 500

    #Build the model
    network = build_model()

    # Cartpole is considered solved if average reward is >= 300 over 100 
    # consecutive trials
    reward_threshold = 250
    running_reward = 0

    # Choose optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    # Keep last episodes reward
    episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)

    with tqdm.trange(max_episodes) as t:
        with tf.GradientTape(persistent = True) as tape:
            for i in t:
                # Define a common loss 
                loss = tf.constant(0.)

                # Start with defining the initial state
                initial_state = tf.constant(env.reset(), dtype=tf.float32)

                episode_reward, loss_step = train_step(initial_state, network, optimizer, gamma, max_steps_per_episode, N)
                episode_reward = int(episode_reward)

                # Update loss
                loss = loss + loss_step/batch_update_size

                episodes_reward.append(episode_reward)
                running_reward = statistics.mean(episodes_reward)

                t.set_description(f'Episode {i}')
                t.set_postfix(episode_reward=episode_reward, running_reward=running_reward)

                # Update network every 'batch_update_size' episodes
                if i % batch_update_size == 0:
                        grads = tape.gradient(loss, network.trainable_variables)
                        optimizer.apply_gradients(zip(grads, network.trainable_variables))
                        loss = tf.constant(0.)
                        del tape

                # Show average episode reward every 10 episodes
                #if i % 10 == 0:
                 #   pass # print(f'Episode {i}: average reward: {avg_reward}')

                #if running_reward > reward_threshold and i >= min_episodes_criterion:  
                 #   break

    #print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}')
    
    return episode_reward
    
cartpole(learning_rate, N, batch_update_size, gamma)
