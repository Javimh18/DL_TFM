import numpy as np
import torch

from agents.dqn_agent import DQNAgent

class DDQNAgent(DQNAgent):
    def __init__(self, 
                 obs_shape: tuple, 
                 action_dim: int, 
                 device: str, 
                 save_net_dir: str, 
                 exp_schedule: str,
                 prioritized_replay: bool,
                 agent_config: dict,
                 nn_config:dict
                 ):
        super().__init__(obs_shape, action_dim, device, save_net_dir, exp_schedule, prioritized_replay, agent_config, nn_config)
        
    @torch.no_grad() # since this is our "ground truth" (look ahead prediction)
    def compute_q_target(self, reward, next_state, done):
        # for the next state, get the actions that have higher q_values
        online_q_action_value = self.net(next_state.float(), model='online')
        max_value_action = torch.argmax(online_q_action_value, dim=1)
        # then, apply those actions onto the target (off-line) model
        target_q_action_values = self.net(next_state.float(), model='target')
        q_next_state_target = target_q_action_values[np.arange(0, self.batch_size), max_value_action]
        
        return (reward + (1-done.float()) * self.gamma * q_next_state_target).float()