import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


# For further information about visualization: 
# https://pythonprogramming.net/q-learning-analysis-reinforcement-learning-python-tutorial/

env = gym.make("MountainCar-v0")

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 40000

SHOW_EVERY = 500

# to make the Q-table manageable, we define buckets that fathers the
# values of intervals from the observation space.
# separate the range of values from the observation space
DISCRETE_OBS_SIZE = [50]*len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OBS_SIZE

epsilon = 0.3
START_EPSILON_DECAY = 1
END_EPSILON_DECAY = EPISODES // 2
epsilon_decay_value = epsilon/(END_EPSILON_DECAY - START_EPSILON_DECAY)

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OBS_SIZE + [env.action_space.n]))
ep_rewards = []
aggr_ep_rewards = {'ep':[], 'avg':[], 'min':[], 'max': []}

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int32))

# looping thorugh episodes
for episode in range(EPISODES):
    episode_reward = 0
    if episode % SHOW_EVERY == 0:
        print("Current episode:", episode)

    discrete_state = get_discrete_state(env.reset()[0])
    done = False
    truncate = False
    while not done and not truncate:
        # epsilon-greedy policy
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)
        new_state, reward, done, truncate, _ = env.step(action)
        episode_reward += reward
        new_discrete_state = get_discrete_state(new_state)
        if not done and not truncate:
            # taking the greedy policy (action that maximizes the q_learning) for the future state
            max_future_q = np.max(q_table[new_discrete_state])
            # taking the e-greedy action for the  current state
            current_q_value = q_table[discrete_state + (action, )]
            # Q(s_t, a_t) <- Q(s_t, a_t) + LR*(r_t+1 + dis*Q(s_t+1, a') - Q(s_t, a_t))
            # where a is the greedy action taken in s_t+1
            new_q = current_q_value + LEARNING_RATE*(reward + DISCOUNT * max_future_q - current_q_value)
            q_table[discrete_state + (action, )] = new_q
        elif new_state[0] > env.unwrapped.goal_position:
            q_table[discrete_state + (action, )] = 0 # if the position reaches the end, then q_value update is 0 (reward by default is -1)

        discrete_state = new_discrete_state # update the current state

    # decreasing the epsilon value as episodes go by to stop exploring and
    # start exploiting
    if END_EPSILON_DECAY >= episode >= START_EPSILON_DECAY:
        epsilon -= epsilon_decay_value

    ep_rewards.append(episode_reward)
    # extract the statistics using the dictionary, extracting the
    # matrics each SHOW_EVERY episodes
    if not episode % SHOW_EVERY:
        np.save(f"q_tables/{episode}-q_table.npy", q_table)
        average_reward = sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))

        print(f"Episode: {episode} | avg: {average_reward} | min: {min(ep_rewards[-SHOW_EVERY:])} \
               | max: {max(ep_rewards[-SHOW_EVERY:])}")

env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label='avg')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label='min')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label='max')
plt.legend(loc=4)
plt.grid(True)
plt.show()

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

