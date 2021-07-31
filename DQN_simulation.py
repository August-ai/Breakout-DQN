# Needed if on google colab
# ! wget http://www.atarimania.com/roms/Roms.rar
# ! mkdir /content/ROM/
# ! unrar e /content/Roms.rar /content/ROM/
# ! python -m atari_py.import_roms /content/ROM/

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import gym
from collections import deque
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Lambda, Input, Add, Subtract
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.initializers import VarianceScaling
import random
import time
import os

def store(state, action, reward, next_state, done):
  memory.append((state, action, reward, next_state, done))
  
def preprocess_frame(frame, shape=(84, 84)):
  frame = frame.astype(np.uint8)
  frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
  frame = frame[34:160+34, :160]
  frame = cv2.resize(frame, (shape[0], shape[1]), interpolation=cv2.INTER_NEAREST)
  return frame
  
def q_model_creation(n_actions, input_shape=(84,84), history_length=4, lr=0.00001):
  model_input = Input(shape=(input_shape[0], input_shape[1], history_length))
  x = Lambda(lambda layer: layer/255)(model_input)
  x = Conv2D(32, (8, 8), strides=4, kernel_initializer=VarianceScaling(scale=2.), activation='relu')(x)
  x = Conv2D(64, (4, 4), strides=2, kernel_initializer=VarianceScaling(scale=2.), activation='relu')(x)
  x = Conv2D(64, (3, 3), strides=1, kernel_initializer=VarianceScaling(scale=2.), activation='relu')(x)
  x = Conv2D(1024, (7, 7), strides=1, kernel_initializer=VarianceScaling(scale=2.), activation='relu')(x)

  val_stream, adv_stream = Lambda(lambda w: tf.split(w, 2, 3))(x)
  val_stream = Flatten()(val_stream)
  val = Dense(1, kernel_initializer=VarianceScaling(scale=2.))(val_stream)
  adv_stream = Flatten()(adv_stream)
  adv = Dense(n_actions, kernel_initializer=VarianceScaling(scale=2.))(adv_stream)

  adv_mean = Lambda(lambda w: tf.reduce_mean(w, axis=1, keepdims=True))(adv)
  q_vals = Add()([val, Subtract()([adv, adv_mean])])

  model = Model(model_input, q_vals)
  model.compile(Adam(lr), loss=tf.keras.losses.Huber())
  return model  
 

def sample_minibatch(batch_size):
  if len(memory) > min_memory:
    index = np.random.randint(0, len(memory)-2-batch_size)
    states = []
    actions = []
    rewards = []
    next_states = []
    terminal_flags = []
    for i in range(batch_size):
      sub_states = np.asarray(memory[index+i][0])
      # change later to more flexible
      sub_states = sub_states.reshape((84, 84, 4))
      sub_next_states = np.asarray(memory[index+i][3])
      sub_next_states = sub_next_states.reshape((84, 84, 4))

      states.append(sub_states)
      next_states.append(sub_next_states)
      actions.append(memory[index+i][1])
      rewards.append(memory[index+i][2])
      terminal_flags.append(memory[index+i][4]) 
    states = np.asarray(states).reshape(batch_size, 84, 84, 4)
    next_states = np.asarray(next_states).reshape(batch_size, 84, 84, 4)
    return index, states, np.asarray(actions), np.asarray(rewards), next_states, np.asarray(terminal_flags)
  return -1

def learn():
  if sample_minibatch(batch_size) == -1:
    return
  index, states, actions, rewards, next_states, terminal_flags = sample_minibatch(batch_size)
  arg_q_max = dqn.predict(next_states).argmax(axis=1)

  # Target DQN estimates q-vals for new states
  future_q_vals = target_dqn.predict(next_states)
  double_q = future_q_vals[range(batch_size), arg_q_max]

  # Calculate targets (bellman equation)
  target_q = rewards + (gamma*double_q * (1-terminal_flags))
  # y = rewards + gamma * np.max(target_dqn.predict(next_states) * (1 - np.asarray(terminal_flags).astype(int)))
  with tf.GradientTape() as tape:
    q_values = dqn(states)

    one_hot_actions = tf.keras.utils.to_categorical(actions, n_actions, dtype=np.float32)  # using tf.one_hot causes strange errors
    Q = tf.reduce_sum(tf.multiply(q_values, one_hot_actions), axis=1)
    error = Q - target_q
    importance = set_priorities(index, error)
    loss = tf.keras.losses.Huber()(target_q, Q)
    loss = tf.reduce_mean(loss * importance)

  model_gradients = tape.gradient(loss, dqn.trainable_variables)
  dqn.optimizer.apply_gradients(zip(model_gradients, dqn.trainable_variables))


def set_priorities(indices, error, offset=0.1):
  if error != None:
    priorities = np.zeros(batch_size)
    for i in range(batch_size):
      priorities[i] = (abs(error[i]) + offset) / (tf.reduce_sum(abs(error)) + offset)
      return priorities

def train():
  env = gym.make('BreakoutDeterministic-v4')
  step = 0
  state_count = 0
  next_state_count = 0
  states_storage = []
  do_store = False
  episode_avg_reward = 0
  total_time = 0
  eps = 1
  show_avg = 50

  for i in range(1, E):
    done = False
    state = env.reset()
    frame = preprocess_frame(state)
    first_frame = True
    lives = 5
    episode_reward = 0
    start_time = time.time()
    storing = False
    storing_step = 0
    inside_step = 0
    while not done:
      action = 1
      if step % 4 == 0 and step > 8:
        if first_frame:
          action = 1
          first_frame = False        
        elif np.random.random() > eps:
          m_states = np.asarray(states_storage[step-4:step])
          # print(m_states.shape)
          m_states = m_states.reshape((1, 84, 84, 4))
          action = 1 + np.argmax(dqn.predict(m_states))
        else:
          action = np.random.randint(1,3)
      next_state, reward, done, info = env.step(action)
      next_frame = preprocess_frame(next_state)
      state = next_frame
      states_storage.append(state)

      do_store = False
      if step > 9 and step % 8 == 0 and np.random.randint(1, 4) == 2:
        try:
          states = np.asarray([s for s in states_storage[step-8:step-4]])
          states = np.reshape(states, (1, 84, 84, 4))
          next_states = np.asarray([ns for ns in states_storage[step-4:step]])
          next_states = np.reshape(next_states, (1, 84, 84, 4))
          do_store = True
        except:
          print('incorrect shape provided', states.shape, next_states.shape)
          do_store = False

      if step > 9 and step % 8 == 0 and do_store:
        store(states, action-1, reward, next_states, done)

      if lives > info['ale.lives']:
        # life lost
        reward = -1
        lives -= 1
        first_frame = True
      episode_reward += reward
      if random.randint(1, 4) == 2:
        learn()
      step += 1
      if inside_step % 10000 == 0 and inside_step > 0:
        done = True
        
      inside_step += 1
    episode_avg_reward += episode_reward
    total_time += time.time() - start_time
    # Epsilon
    if step % target_update_frequency == 0 and step > 0:
        target_dqn.set_weights(dqn.get_weights())
    if i > 150:
      eps = eps * eps_decay
      eps = max(eps, eps_min)
    if i % show_avg == 0:
      print(f"Episode: {i}; total_reward: {episode_avg_reward / show_avg}; steps:{step} epsilon: {round(eps,3)}, finished in: {round(total_time,2)} seconds")
      print(len(memory))
      total_time = 0
      episode_avg_reward = 0
    if i % 500 == 0:
      dqn.save('model_saved.h5')

# INITIALIZE VARIABLES
memory = deque(maxlen=100_000)
min_memory = 2000
gamma = 0.99
batch_size = 64
n_actions = 3
eps = 1
eps_decay = 0.9995
eps_min = 0.01
E = 100_000
target_update_frequency = 500

dqn = q_model_creation(3)
target_dqn = q_model_creation(3)

train()
