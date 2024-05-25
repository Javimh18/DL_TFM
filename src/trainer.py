import gymnasium as gym
from gym.utils.save_video import save_video

from utils.logger import MetricLogger
from agents.ddqn_agent import DDQNAgent
from agents.dqn_agent import DQNAgent

class Trainer:
    """
    Handler for training the Agent
    """
    def __init__(self, 
                 env: gym.Env, 
                 agent:DQNAgent|DDQNAgent, 
                 n_steps: int, 
                 log_every=200, 
                 save_check_dir="../checkpoint",
                 save_video_dir='../video'):
        self.env = env
        self.agent = agent
        self.n_steps = n_steps
        self.curr_step = 0
        self.curr_episode = 0
        self.log_every = log_every
        self.save_check_dir = save_check_dir
        self.save_video_dir = save_video_dir
        
    def train(self):
        """
        This function implements the training loop of an experience replay DQN agent.
        """
        # create if not exists
        self.save_check_dir.mkdir(parents=True)
        self.save_video_dir.mkdir(parents=True)
        logger = MetricLogger(self.save_check_dir)
    
        while self.curr_step < self.n_steps:
            # reset environment
            done, trunc = False, False
            state = self.env.reset()
            #measure_array = []
            while (not done) and (not trunc):
                # 1. get action for state
                action = self.agent.perform_action(state) # 20.69 ms  
                # 2. run action in environment
                next_state, reward, done, trunc, info = self.env.step(action) # 1.70 ms
                # 3. collect experience in exp. replay buffer for Q-learning
                self.agent.store_transition(state, action, reward, next_state, done, trunc) # 0.56 ms
                # 4. Learn from collected experiences
                #start = datetime.datetime.now()
                q, loss = self.agent.learn(self.curr_step) # 39.76 ms
                #measure = datetime.datetime.now() - start
                #measure = measure.total_seconds() * 1000
                # 5. Update the current state 
                state = next_state
                # 6. Update step value 
                self.curr_step += 1            
                #measure_array.append(measure)
                logger.log_step(loss, q)
            
            if 'episode' in info:
                # episode field is stored in the info dict if episode ended
                logger.log_episode(ep_length=info['episode']['l'], ep_reward=info['episode']['r'],)
                if not(self.curr_episode % self.log_every) :
                    logger.record(episode=self.curr_episode, 
                                  epsilon=self.agent.exploration_rate, 
                                  step=self.curr_step)
                # log the real reward using episode statistics
                self.curr_episode += 1
            
            # avg_measure = sum(measure_array)/len(measure_array)
            # print(f"Avg. step time for measure: {avg_measure:.2f} ms")
            
            """
            if self.curr_episode > 999: 
                save_video(
                    self.env.render(),
                    self.save_video_dir,
                    fps=self.env.metadata["render_fps"],
                    episode_index=self.curr_episode
                )
            """
                
                
            
        
        
        