import numpy as np
import torch

from agents.dqn_agent import DQNAgent, REPLAY_MEMORY_SIZE, TRANSITION_KEYS

class DDQNAgent(DQNAgent):
    def __init__(self, 
                 obs_shape: tuple, 
                 action_dim: int, 
                 device: str, 
                 save_net_dir: str, 
                 agent_config: dict,
                 nn_config:dict
                 ):
        super().__init__(obs_shape, action_dim, device, save_net_dir, agent_config, nn_config)
        
    def learn(self, step):
        # Once the error is computed, each sync_every the weights of the 
        # target network are updated to the online network
        if not (step % self.sync_every):
            self.sync_Q_target()
            
        # save the model each save_every steps
        if step % self.save_every == 0 and step > 0:
            self.save(step)
        
        # Burning lets episodes pass w/o doing nothing?
        if step < self.burning:
            return None, None

        # Not learning every step, but every learn_every steps
        if step % self.learn_every != 0:
            return None, None
        
        # sample from memory 
        transitions_batch = self.memory.sample(self.batch_size)
        # extract samples such that: s_t, a_t, r_t+1, s_t+1
        state, action, reward, next_state, done, trunc = (transitions_batch.get(key) for key in \
                                                    (TRANSITION_KEYS))
        
        # since they are all tensors, we only need to squeeze the ones with an additional dimension
        state, action, reward, next_state, done, trunc = \
            state, action.squeeze(), reward.squeeze(), next_state, done.squeeze(), trunc.squeeze()
    
        # once we have our transition tuple, we apply TD learning over our DDQN and compute the loss 
        # between the targets (the look ahead) and the estimations (what we know for the current state)
        self.net.train()
        q_estimate = self.compute_q_estimate(state, action)
        q_target = self.compute_q_target(reward, next_state, done)
        
        self.optimizer.zero_grad()
        loss = self.loss_fn(q_target, q_estimate)
        loss.backward()
        self.optimizer.step()
        
        return q_estimate.mean().item(), loss.item()
    
    def compute_q_estimate(self, state, action):
        # get the q values estimates for the states || Q_online(s,a;w)
        q_estimates = self.net(state, model='online')
        # select, for each q_estimate, the q_value of the selected action
        q_action_estimates = q_estimates[np.arange(0, self.batch_size), action]
        return q_action_estimates
        
    @torch.no_grad() # since this is our "ground truth" (look ahead prediction)
    def compute_q_target(self, reward, next_state, done):
        # for the next state, get the actions that have higher q_values
        online_q_action_values = self.net(next_state, model='online')
        max_value_actions = torch.argmax(online_q_action_values, dim=-1)
        # then, apply those actions onto the target (off-line) model
        target_q_action_values = self.net(next_state, model='target')
        q_next_state_target = target_q_action_values[np.arange(0, self.batch_size), max_value_actions]
        
        return (reward + (1-done.float())*self.gamma*q_next_state_target).float()