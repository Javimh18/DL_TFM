import gymnasium as gym
from gym.utils.save_video import save_video
import pandas as pd
from pathlib import Path
import datetime

from utils.logger import MetricLogger
from agents.ddqn_agent import DDQNAgent
from agents.dqn_agent import DQNAgent
from utils.schedulers import PowerDecayScheduler

class Trainer:
    """
    Handler for training the Agent
    """
    def __init__(self, 
                 env: gym.Env, 
                 agent:DQNAgent|DDQNAgent, 
                 n_steps: int, 
                 logger:MetricLogger,
                 log_every=200, 
                 save_check_dir="../checkpoint",
                 save_video_dir='../video',
                 save_video_progress:bool=False):
        self.env = env
        self.agent = agent
        self.n_steps = n_steps
        self.curr_step = 0
        self.curr_episode = 0
        self.log_every = log_every
        self.save_check_dir = save_check_dir
        self.save_video_dir = save_video_dir
        self.save_vid_flag = save_video_progress
        self.logger = logger
        
    def train(self):
        """
        This function implements the training loop of an experience replay DQN agent.
        """
    
        # only save videos in training if specified
        if self.save_vid_flag:
            self.save_video_dir.mkdir(parents=True)
    
        while self.curr_step < self.n_steps:
            # reset environment
            done, trunc = False, False
            state = self.env.reset()
            measure_array = []
            while (not done) and (not trunc):
                # 1. get action for state
                action = self.agent.perform_action(state, self.curr_step) # 20.69 ms  
                # 2. run action in environment
                start = datetime.datetime.now()
                next_state, reward, done, trunc, info = self.env.step(action) # 1.70 ms
                measure = datetime.datetime.now() - start
                # 3. collect experience in exp. replay buffer for Q-learning
                self.agent.store_transition(state, action, reward, next_state, done) # 0.56 ms
                # 4. Learn from collected experiences
                q, loss = self.agent.learn(self.curr_step) # 39.76 ms
                measure = measure.total_seconds() * 1000
                measure_array.append(measure)
                # 5. Update the current state 
                state = next_state
                # 6. Update step value 
                self.curr_step += 1            
                self.logger.log_step(loss, q)
            
            # since we are dealing with an episodic life env, at the end of each episode
            # the info dictionary contains the relevant statistics for the reward and length
            if 'episode' in info:
                # episode field is stored in the info dict if episode ended
                self.logger.log_episode(ep_length=info['episode']['l'], ep_reward=info['episode']['r'],)
                if not(self.curr_episode % self.log_every) :
                    self.logger.record(episode=self.curr_episode, 
                                  epsilon=self.agent.exploration_rate, 
                                  step=self.curr_step)
                # log the real reward using episode statistics
                self.curr_episode += 1
            
            # avg_measure = sum(measure_array)/len(measure_array)
            # print(f"Avg. step time for measure: {avg_measure:.2f} ms")
            
            if self.save_vid_flag and self.curr_episode > 999: 
                save_video(
                    self.env.render(),
                    self.save_video_dir,
                    fps=self.env.metadata["render_fps"],
                    episode_index=self.curr_episode
                )
                
    def load_prev_training_info(self, training_folder:Path):
        log_prev_training = training_folder / "log"
        df = pd.read_csv(log_prev_training, header=0, sep='\s+', skipinitialspace=True)
        max_step = max(df['Step'].values)
        max_ep = max(df['Episode'].values)
        self.curr_episode = max_ep
        self.curr_step = max_step
        
        # in case the scheduler is of type powerdecay, we
        # set the scheduler to the state that it was in the
        # prev training
        if type(self.agent.exp_scheduler) == PowerDecayScheduler:
            for t in range(max_step):
                self.agent.exp_scheduler.step(t)
                