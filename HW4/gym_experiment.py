from HW4.gym import gym

'''
env = gym.make('LunarLander-v2')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())

'''

game0 = 'CartPole-v0'
game1 = 'LunarLander-v2'
import gym
env = gym.make(game1)
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break