import torch
from torch import nn
from collections import deque
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from torchrl.data.replay_buffers.samplers import PrioritizedSampler
from tensordict import TensorDict
import numpy as np
import datetime
from utils.utils import first_if_tuple
from agents.dqn_models import DQN

TRANSITION_KEYS = ("state", "action", "reward", "next_state", "done", 'trunc')

class DQNAgent:
    def __init__(self, 
                 obs_shape:tuple,
                 action_dim: int, 
                 device: str, 
                 save_net_dir: str,
                 agent_config: dict,
                 nn_config:dict):
        
        self.device = device
        self.action_dim = action_dim
        self.obs_shape = obs_shape
        self.type = agent_config['type']
        
        # defining the DQN
        self.net = DQN(type=self.type, 
                       n_actions=self.action_dim, 
                       obs_shape=self.obs_shape,
                       config=nn_config).float()
        self.net = self.net.to(self.device)
        # defining the memory (experience replay) of the agent
        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(
            max_size=float(agent_config['replay_memory_size']),
            scratch_dir='./memmap_dir',
            device=self.device,
        ),sampler=PrioritizedSampler(max_capacity=int(float(agent_config['replay_memory_size'])), 
          alpha=1.0, 
          beta=1.0))
        
        # hyperparameters
        self.batch_size = int(agent_config["batch_size"])
        self.gamma = float(agent_config["gamma"] )
        self.lr = float(agent_config['lr'])
        
        # loss function and optimizer
        self.optimizer = torch.optim.Adamax(self.net.parameters(), lr=self.lr)
        self.loss_fn = torch.nn.SmoothL1Loss()
        
        # exploration (epsilon) parameter for e-greedy policy
        self.exploration_rate = float(agent_config['exp_rate_max'])
        self.exploration_rate_min = float(agent_config['exp_rate_min'])
        self.exploration_rate_decay = \
            (self.exploration_rate_min/self.exploration_rate)**(1/float(agent_config['steps_to_explore']))
        
        # learn and burning parameters
        self.learn_every = int(agent_config['learn_every'])
        self.burning = float(agent_config['burning'])
        
        # update the q_target each sync_every steps
        self.sync_every = float(agent_config["sync_every"])
        self.save_every = float(agent_config["save_every"])
        self.save_net_dir = save_net_dir
    
    
    @torch.no_grad()
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
        
        return action
        
        
    def store_transition(self, state, action, reward, next_state, done, trunc):
        # check if the environment returned the state as a tuple
        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()
        # cast the information from the env.step() as a tensor
        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
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
    
    
    def compute_q_estimate(self, state, action):
        # get the q values estimates for the states || Q_online(s,a;w)
        q_estimates = self.net(state, model='online')
        # select, for each q_estimate, the q_value of the selected action
        # np.arange does the trick of extracting all the q_values from the batch, given the action
        q_action_estimates = q_estimates[np.arange(0, self.batch_size), action]
        return q_action_estimates
    
    
    @torch.no_grad() # since this is our "ground truth" (look ahead prediction)
    def compute_q_target(self, reward, done, next_state):
        q_max_value, _ = torch.max(self.net(next_state, model='target'), dim=1)
        return reward + (1 - done.float()) * self.gamma * q_max_value
    
    
    def recall(self):
        # sample from memory 
        transitions_batch = self.memory.sample(self.batch_size)
        # extract samples such that: s_t, a_t, r_t+1, s_t+1
        state, action, reward, next_state, done, trunc = (transitions_batch.get(key) for key in \
                                                    (TRANSITION_KEYS))
        return state, action.squeeze(), reward.squeeze(), next_state, done.squeeze(), trunc.squeeze()
        
        
    def learn(self, step):
        
        # Once the error is computed, each sync_every the weights of the 
        # target network are updated to the online network
        if not (step % self.sync_every):
            self.sync_Q_target()
            
        # save the model each save_every steps
        if step % self.save_every == 0 and step > 0:
            self.save(step)
        
        # Burning lets episodes pass but collects experiences for the memory buffer
        if step < self.burning:
            return None, None

        # Not learning every step, but every "learn_every" steps
        if step % self.learn_every != 0:
            return None, None
        
        state, action, reward, next_state, done, _ = self.recall()
            
        # once we have our transition tuple, we apply TD learning over our DQN and compute the loss
        q_estimate = self.compute_q_estimate(state, action)
        q_target = self.compute_q_target(reward, done, next_state)
            
        # update the q_online network using the loss
        loss = self.loss_fn(q_target, q_estimate) # Compute Huber loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
            
        return q_estimate.mean().item(), loss.item()
    
    
    def sync_Q_target(self):
        """
        Transferring the parameters from the online net to the target net.
        """
        self.net.target.load_state_dict(self.net.online.state_dict())
        
        
    def save(self, step:int):
        save_path = (
            self.save_net_dir / f"{self.type}_net_{int(step // self.save_every)}_{self.gamma}_{self.lr}.chkpt"
        )
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate, type_model=self.type),
            save_path,
        )
        print(f"INFO: DQNAgent net saved to {save_path} at step {step}")
        
        
    def load_weights(self, path_to_checkpoint:str, set_epsilon:bool):
        checkpoint = torch.load(path_to_checkpoint)
        
        if checkpoint['type'] != self.type:
            print(f"ERROR: Tried to load a net with a different type."
                  f"Declared model: {self.type} | Loaded model: {checkpoint['type']}"
                  f"Please review the datatypes."
                  f"Exiting...")
            exit()

        # load model from dictionary
        self.net.load_state_dict(checkpoint['model'])
        # set the exploration rate to whatever it was trained
        if set_epsilon:
            print(f"INFO: Saved exploration rate loaded. eps_max={checkpoint['exploration_rate']}")
            self.exploration_rate = checkpoint['exploration_rate']
        else:
            print(f"INFO: Saved exploration rate not loaded. eps_max={self.exploration_rate}") 

