import torch
from torch import nn
from torch.nn import Sequential
from collections import deque
import random

import gymnasium as gym

REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 64
EPISODES = 2000

class DQNAgent(nn.Module):
    def __init__(self, state_dim, action_dim) -> None:
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim

        # gets trained every step
        self.online = self.__build_cnn(state_dim, action_dim)
        # each n steps it updates the values from the online model
        # for stability purposes
        self.target = self.__build_cnn(state_dim, action_dim)
        self.target.load_state_dict(self.online.state_dict())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.update_counter = 0

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Q_target network parameters are frozen
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)
        
    def update_memory(self, transition):
        self.replay_memory.append(transition)
        
    def __build_cnn(self, input_dim, output_dim):

        c, h, w = input_dim
        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input height: 84, got: {w}")
        
        return nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
    
    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
           return

        minibatch = random.sample(self.replay_memory, MIN_REPLAY_MEMORY_SIZE)
       
def first_if_tuple(x):
    return x[0] if isinstance(x, tuple) else x 
        
if __name__ == '__main__':
    
    env = gym.make("ALE/Tennis-v5", obs_type='rgb')
    state = first_if_tuple(env.reset())
    agent = DQNAgent(state_dim=(state.shape), action_dim=env.action_space.n)
    for episode in range(EPISODES):
        next_step, reward, done, trunc, info = env.step(action=0)

