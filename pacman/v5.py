import numpy as np
import gym
import argparse
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
from utils.utils import get_last_weights_filepath

INPUT_SHAPE = (84,84)
WINDOW_LENGTH = 4

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()

class AtariProcessor(Processor):
  def process_observation(self, observation):
    assert observation.ndim == 3
    img = Image.fromarray(observation)
    img = img.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
    processed_observation = np.array(img)
    assert processed_observation.shape == INPUT_SHAPE
    return processed_observation.astype('uint8')  # saves storage in experience memory

  def process_state_batch(self, batch):
    processed_batch = batch.astype('float32') / 255.
    return processed_batch

  def process_reward(self, reward):
    return np.clip(reward, -1., 1.)

env = gym.make('MsPacman-v0')
np.random.seed(231)
env.seed(123)
nb_actions = env.action_space.n
print("NUMBER OF ACTIONS: {}".format(nb_actions))

input_shape = (WINDOW_LENGTH, INPUT_SHAPE[0], INPUT_SHAPE[1])
frame = Input(shape=(input_shape))
cv1 = Convolution2D(32, kernel_size=(8,8), strides=4, activation='relu', data_format='channels_first')(frame)
cv2 = Convolution2D(64, kernel_size=(4,4), strides=2, activation='relu', data_format='channels_first')(cv1)
cv3 = Convolution2D(64, kernel_size=(3,3), strides=1, activation='relu', data_format='channels_first')(cv2)
dense = Flatten()(cv3)
dense = Dense(512, activation='relu')(dense)
buttons = Dense(nb_actions, activation='linear')(dense)
model = Model(inputs=frame, outputs=buttons)
model.summary()

memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
processor = AtariProcessor()
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05, nb_steps=1250000)

dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
               processor=processor, enable_double_dqn=False, enable_dueling_network=False, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
               train_interval=4, delta_clip=1.)
dqn.compile(Adam(lr=.00025), metrics=['mae'])

folder_path = './model_saves/MsPacmanv5/'

if args.mode == 'train':
    weights_filename = folder_path + 'dqn_pacman_weights.h5f'
    checkpoint_weights_filename = folder_path + 'dqn_pacman_weights_{step}.h5f'
    log_filename = folder_path + 'dqn_pacman_REWARD_DATA.txt'
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=500000)]
    # callbacks += [TrainEpisodeLogger.on_action_begin(log_filename)]
    dqn.fit(env, callbacks=callbacks, nb_steps=10000000, verbose=0, nb_max_episode_steps=20000)

elif args.mode == 'test':
    # weights_filename = get_last_weights_filepath(folder_path)

    weights_filename = folder_path+"dqn_MsPacman-v0_weights_100000.h5f"
    dqn.load_weights(weights_filename)
    dqn.test(env, nb_episodes=10, visualize=True, nb_max_start_steps=80)