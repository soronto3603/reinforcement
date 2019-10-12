import gym
env = gym.make('CartPole-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        # action = env.action_space.sample()
        action = 0
        print("action : {}".format(action))
        observation, reward, done, info = env.step(action)
        print("observation : {}".format(observation))
        print("reward : {}".format(reward))
        print(info)
        print("done : {}".format(done))
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()