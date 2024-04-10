import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")
env.reset()

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 5000

SHOW_EVERY = 500

# to make the Q-table manageable, we define buckets that fathers the
# values of intervals from the observation space.
# separate the range of values from the observation space
DISCRETE_OBS_SIZE = [20]*len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OBS_SIZE

epsilon = 0.2
START_EPSILON_DECAY = 1
END_EPSILON_DECAY = EPISODES // 2
epsilon_decay_value = epsilon/(END_EPSILON_DECAY - START_EPSILON_DECAY)

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OBS_SIZE + [env.action_space.n]))

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int32))

for episode in range(EPISODES):
    if episode % SHOW_EVERY == 0:
        print("Current episode:", episode)

    discrete_state = get_discrete_state(env.reset()[0])
    done = False
    truncate = False
    while not done and not truncate:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)
        new_state, reward, done, truncate, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)
        if not done and not truncate:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q_value = q_table[discrete_state + (action, )]
            new_q = (1 - LEARNING_RATE)*current_q_value + LEARNING_RATE*(reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action, )] = new_q
        elif new_state[0] > env.unwrapped.goal_position:
            print(f"The problem is solved in episode: {episode}")
            q_table[discrete_state + (action, )] = 0 # if the position reaches the end, then q_value update is 0 (reward by default is -1)

        discrete_state = new_discrete_state # update the current state

    # decreasing the epsilon value as episodes go by to stop exploring and
    # start exploiting
    if END_EPSILON_DECAY >= episode >= START_EPSILON_DECAY:
        epsilon -= epsilon_decay_value

# Once training has ended
env = gym.make("MountainCar-v0", render_mode='human')
discrete_state = get_discrete_state(env.reset()[0])
done = False
truncate = False
while not done and not truncate:
    action = np.argmax(q_table[discrete_state])
    new_state, reward, done, truncate, _ = env.step(action)
    new_discrete_state = get_discrete_state(new_state)
    discrete_state = new_discrete_state

    
env.close()

