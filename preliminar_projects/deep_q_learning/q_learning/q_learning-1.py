import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0", render_mode='human')
env.reset()

print(env.observation_space.high)
print(env.observation_space.low)
print(env.action_space.n)

# to make the Q-table manageable, we define buckets that fathers the
# values of intervals from the observation space.
# separate the range of values from the observation space
DISCRETE_OBS_SIZE = [20]*len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OBS_SIZE

print(discrete_os_win_size)

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OBS_SIZE + [env.action_space.n]))
print(q_table.shape)

done = False

while not done:
    action = 2
    new_state, reward, done, _, _ = env.step(action)
    print(reward, new_state)
    env.render()
    
env.close()

