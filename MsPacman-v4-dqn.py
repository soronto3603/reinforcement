
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import random
import gym

from collections import deque
from time import sleep

print(tf.__version__)

def bot_play(model: tf.keras.Model) -> None:
    state = env.reset()
    total_reward = 0

    while True:
      env.render()
      action = np.argmax(model.predict(state))
      state, reward, done, _ = env.step(action)
      total_reward += reward
      if done:
          print("Total score: {}".format(total_reward))
          break

def annealing_epsilon(episode: int, min_e: float, max_e: float, target_episode: int) -> float:
    slope = (min_e - max_e) / (target_episode)
    intercept = max_e

    return max(min_e, slope * episode + intercept)
def make_model() -> tf.keras.Model:
  keras = tf.keras
  layers = keras.layers

  conv_input = layers.Input(shape=(210,160,3))
  conv = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(conv_input)
  conv = layers.Conv2D(64, (3, 3), activation='relu')(conv)
  conv = layers.MaxPool2D(pool_size=(2, 2))(conv)
  conv = layers.Flatten()(conv)
  conv = layers.Dense(128, activation='relu')(conv)
  conv = layers.Dense(9, activation='softmax')(conv)

  model = keras.models.Model(inputs= [conv_input], outputs=conv)
  
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  
  return model
def update(model: tf.keras.Model, x, y) -> float:
  return model.fit(x=x,y=y).history['loss']

def predict(model: tf.keras.Model, x):
  return model.predict(x)
def predictOne(model: tf.keras.Model, x):
  return model.predict(np.array([x]))
def train_minibatch(model: tf.keras.Model, train_batch: list) -> float:
    state_array = np.array([x[0] for x in train_batch])
    action_array = np.array([x[1] for x in train_batch])
    reward_array = np.array([x[2] for x in train_batch])
    next_state_array = np.array([x[3] for x in train_batch])
    done_array = np.array([x[4] for x in train_batch])

    X_batch = state_array
    y_batch = model.predict(state_array)
    Q_target = reward_array + DISCOUNT_RATE * np.max(model.predict(next_state_array), axis=1) * ~done_array
    y_batch[np.arange(len(X_batch)), action_array] = Q_target
    loss = update(model, X_batch, y_batch)

    return loss

model = make_model()
env = gym.make('MsPacman-v0')
env = gym.wrappers.Monitor(env, 'gym-results/pacman/', force=True)
INPUT_SIZE = env.observation_space.shape[0]
OUTPUT_SIZE = env.action_space.n

DISCOUNT_RATE = 0.99
REPLAY_MEMORY = 50000
MAX_EPISODE = 5000
BATCH_SIZE = 4

# minimum epsilon for epsilon greedy
MIN_E = 0.0
# epsilon will be `MIN_E` at `EPSILON_DECAYING_EPISODE`
EPSILON_DECAYING_EPISODE = MAX_EPISODE * 0.01

def main(): 
  replay_buffer = deque(maxlen=REPLAY_MEMORY)
  last_100_game_reward = deque(maxlen=100)

  for episode in range(MAX_EPISODE):
    e = annealing_epsilon(episode, MIN_E, 1.0, EPSILON_DECAYING_EPISODE)
    done =False
    state = env.reset()

    step_count = 0
    while not done:
      env.render()
      if np.random.rand() < e:
        action = env.action_space.sample()
      else:
        action = np.argmax(predictOne(model,state))
      
      next_state, reward, done, _ = env.step(action)
      print("action : {}, reward : {}".format(action,reward))
      print("[Episode {:>5}]  steps: {:>5} e: {:>5.2f}".format(episode, step_count, e))
      replay_buffer.append((state, action, reward, next_state, done))

      state = next_state
      step_count += 1

      if len(replay_buffer) > BATCH_SIZE and step_count % 5 == 1:
        minibatch = random.sample(replay_buffer, BATCH_SIZE)
        train_minibatch(model, minibatch)
    print("[Episode {:>5}]  steps: {:>5} e: {:>5.2f}".format(episode, step_count, e))

    last_100_game_reward.append(step_count)
    if len(last_100_game_reward) == last_100_game_reward.maxlen:
      avg_reward = np.mean(last_100_game_reward)
      if avg_reward > 1000:
        print("Game Cleared within {} episodes with avg reward {}".format(episode, avg_reward))
        break
  bot_play(model)

if __name__ == "__main__":
  main()