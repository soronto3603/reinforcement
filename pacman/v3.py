import gym
import numpy as np
import tensorflow as tf
import random
import dqn
from time import sleep

from collections import deque

# ACTION_MEANING = {
#     0: "NOOP",
#     1: "FIRE",
#     2: "UP",
#     3: "RIGHT",
#     4: "LEFT",
#     5: "DOWN",
#     6: "UPRIGHT",
#     7: "UPLEFT",
#     8: "DOWNRIGHT",
#     9: "DOWNLEFT",
#     10: "UPFIRE",
#     11: "RIGHTFIRE",
#     12: "LEFTFIRE",
#     13: "DOWNFIRE",
#     14: "UPRIGHTFIRE",
#     15: "UPLEFTFIRE",
#     16: "DOWNRIGHTFIRE",
#     17: "DOWNLEFTFIRE",
# }

env = gym.make('MsPacman-v0')
env = gym.wrappers.Monitor(env, 'gym-results/pacman/', force=True)
for i_episode in range(20):
    observation = env.reset()
    t=0
    while(True):
        env.render()
        action = random.randrange(0,8)
        observation, reward, done, info = env.step(action)
        print(reward,done,info)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            t=t+1
            break
env.close()

