# the environment has 3 blobs is a grid:
# 1. player
# 2. reward
# 3. enemy
# The idea is for the player (agent) to reach food and eat it.
# The enemy will try to prevent this from happening. 
# The enemy at first will not move, but then, we will make the enemy
# move in order to teach the agent to not be close to the enemy.

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from matplotlib import style
import time
import pickle

style.use('ggplot')

SIZE = 10
HM_EPISODES = 25_000
MOVE_PENALTY = 1
ENEMY_PENALTY = 300
FOOD_REWARD = 25
epsilon = 0.9
EPS_DECAY = 0.9998
SHOW_EVERY = 3000

start_q_table = None # or filename that we want to import

LEARNING_RATE = 0.1
DISCOUNT = 0.95

PLAYER_N = 1
FOOD_N = 2
ENEMY_N = 3

N_STEPS = 200

d = {1: (255,175,0),
     2: (0, 255, 0),
     3: (0,0,255) }

class Blob:
    def __init__(self):
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)

    def __str__(self) -> str:
        return f"{self.x}, {self.y}"
    
    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)

    def action(self, choice):
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)

    def move(self, x=False, y=False):
        if not x:
            self.x += np.random.randint(-1,2)
        else:
            self.x += x

        if not y:
            self.y += np.random.randint(-1,2)
        else:
            self.y += y

        # ERROR CONTROL if Blob tries to go out of the map
        if self.x < 0:
            self.x = 0
        elif self.x > SIZE-1:
            self.x = SIZE - 1

        if self.y < 0:
            self.y = 0
        elif self.y > SIZE-1:
            self.y = SIZE - 1

# initializing all possible combination of the observation space.
# our observation space is going to be (x1, y1) that tell the distance
# to the food, and (x2, y2) that tells the distance to the enemy
if start_q_table is None:
    q_table = {}
    for i in range(-SIZE+1, SIZE):
        for j in range(-SIZE+1, SIZE):
            for k in range(-SIZE+1, SIZE):
                for l in range(-SIZE+1, SIZE):
                    q_table[((i, j), (k, l))] = [np.random.uniform(-5, 0) for i in range(4)]
else:
    with open(start_q_table, 'rb') as f:
        q_table = pickle.load(f)                   
        
episode_rewards = [0]
for episode in range(HM_EPISODES):
    player = Blob()
    food = Blob()
    enemy = Blob()

    if episode % SHOW_EVERY == 0:
        print(f"on # {episode}, epsilon: {epsilon}")
        print(f"{episode} ep mean {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    episode_reward = 0
    for i in range(N_STEPS):
        obs = (player-food, player-enemy)
        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0,4)

        player.action(action)

        # if we end up in the enemy position
        if player.x == enemy.x and player.y == enemy.y:
            reward = -ENEMY_PENALTY
        elif player.x == food.x and player.x == food.y:
            reward = FOOD_REWARD
        else:
            reward = -MOVE_PENALTY

        new_obs = (player-food, player-enemy)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[new_obs][action]

        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        elif reward == ENEMY_PENALTY:
            new_q = -ENEMY_PENALTY
        else:
            new_q = current_q + LEARNING_RATE*(reward + DISCOUNT * max_future_q - current_q)

        q_table[obs][action] = new_q

        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            env[food.y][food.x] = d[FOOD_N]
            env[player.y][player.x] = d[PLAYER_N]
            env[enemy.y][enemy.x] = d[ENEMY_N]

            img = Image.fromarray(env, "RGB")
            img = img.resize((300,300))
            cv2.imshow("", np.array(img))
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
                if cv2.waitKey(500) & 0xFF == ord("q"):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        episode_reward += reward
        if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
            break

    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"reward")
plt.ylabel(f"episode #")
plt.show()

with open(f"q_table-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)