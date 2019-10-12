from env2048.env2048 import Env2048

import numpy as np
import gym
from keras.models import Model
from keras.layers import Flatten, Convolution2D, Input, Dense
from keras.optimizers import Adam
import keras.backend as K
from PIL import Image
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import TrainEpisodeLogger, ModelIntervalCheckpoint

env = Env2048()
np.random.seed(231)
nb_actions = 4
print("NUMBER OFACTIONS: {}".format(nb_actions))

input_shape = [4,4,4]
frame = Input(shape=(input_shape))
x = Flatten()(frame)
x = Dense(32, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(128, activation='relu')(x)
output = Dense(nb_actions, activation='linear')(x)
model = Model(inputs=frame, outputs=output)
model.summary()

memory = SequentialMemory(limit=1000, window_length=4)
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05, nb_steps=1250000)
dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
              enable_double_dqn=False, enable_dueling_network=False, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
               train_interval=4, delta_clip=1.)

dqn.compile(Adam(lr=.00025), metrics=['mae'])

folder_path = './model_saves/env2048/'
weights_filename = '{}env2048-v0_weights.h5f'.format(folder_path)
checkpoint_weights_filename = './model_saves/Vanilla/dqn_env2048-v0_weights_{step}.h5f'
log_filename = folder_path + 'dqn_env2048-v0_REWARD_DATA.txt'
callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=500000)] 
# callbacks += [TrainEpisodeLogger(log_filename)]
dqn.fit(env, callbacks=callbacks, visualize=False ,nb_steps=10000000, verbose=2, nb_max_episode_steps=20000)