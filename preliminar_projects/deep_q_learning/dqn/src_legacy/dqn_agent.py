import torch
from torch import nn
from collections import deque
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from tensordict import TensorDict
import numpy as np

from utils import first_if_tuple

REPLAY_MEMORY_SIZE = 50_000
TRANSITION_KEYS = ("state", "action", "reward", "next_state", "done", 'trunc')

class Agent:
    def __init__(self, state_dim, action_dim, device, batch_size, sync_every=1e4, gamma=0.9) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # defining the DQN
        self.net = DQN(self.state_dim, self.action_dim).float()
        self.net = self.net.to(self.device)
        # defining the memory (experience replay) of the agent
        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(
            max_size=REPLAY_MEMORY_SIZE,
            scratch_dir='./memmap_dir',
            device=self.device
        ))
        
        self.curr_step = 0
        
        # hyperparameters
        self.batch_size = batch_size
        self.gamma = gamma 
        
        # loss function and optimizer
        self.optimizer = torch.optim.Adamax(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()
        
        # exploration (epsilon) parameter for e-greedy policy
        self.exploration_rate = 0.5
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        
        # update the q_target each sync_every steps
        self.sync_every = sync_every
        self.burnin = 1e4  # min. experiences before training
        self.learn_every = 3  # no. of experiences between updates to Q_online
        
    def perform_action(self, state):
        # decide wether to exploit or explore
        if np.random.random() < self.exploration_rate:
            action =  np.random.randint(0, self.action_dim)
        else:
            # use of __array__(): https://gymnasium.farama.org/main/_modules/gymnasium/wrappers/frame_stack/
            state = first_if_tuple(state).__array__()
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            q_values = self.net(state, model='online')
            action = torch.argmax(q_values, dim=1).item()
        
         # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        self.curr_step += 1
        
        return action
        
    def store_transition(self, state, action, reward, next_state, done, trunc):
        # check if the environment returned the state as a tuple
        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()
        # cast the information from the env.step() as a tensor
        state = np.array(torch.tensor(state))
        next_state = np.array(torch.tensor(next_state))
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        done = torch.tensor([done])
        trunc = torch.tensor([trunc])
        # add it to the experience replay buffer
        self.memory.add(TensorDict({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'trunc': trunc
        }, batch_size=[]))
        
    def learn(self, step):
        
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None
        
        # sample from memory 
        transitions_batch = self.memory.sample(self.batch_size).to(self.device)
        # extract samples such that: s_t, a_t, r_t+1, s_t+1
        state, action, reward, next_state, done, trunc = (transitions_batch.get(key) for key in \
                                                    (TRANSITION_KEYS))
        
        # since they are all tensors, we only need to squeeze the ones with an additional dimension
        state, action, reward, next_state, done, trunc = \
            state, action.squeeze(), reward.squeeze(), next_state, done.squeeze(), trunc.squeeze()
            
        # once we have our transition tuple, we apply TD learning over our DQN and compute the loss
        q_estimate = self.net(state, model='online')[np.arange(0, self.batch_size),action]
        # (1-(done and trunc))
        q_target = reward + (1 - done.float())*self.gamma*torch.max(self.net(next_state, model='target'))
        loss = self.loss_fn(q_estimate, q_target)
                    
        # Optimize using Adamax (we can use SGD too)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Once the error is computed, each sync_every the weights of the 
        # target network are updated to the online network
        if not (step % self.sync_every):
            self.sync_Q_target()
            
        return q_estimate.mean().item(), loss.item()
    
    def sync_Q_target(self):
        """
        Transferring the parameters from the online net to the target net.
        """
        self.net.target.load_state_dict(self.net.online.state_dict())
         
        
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim) -> None:
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.online = self.__build_cnn(state_dim, action_dim)
        self.target = self.__build_cnn(state_dim, action_dim)
        self.target.load_state_dict(self.online.state_dict())

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Q_target network parameters are frozen
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)
        
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
        

