import gymnasium as gym
from gym.utils.save_video import save_video

from utils.logger import MetricLogger
import datetime

class Trainer:
    def __init__(self, 
                 env: gym.Env, 
                 agent, 
                 n_episodes: int, 
                 log_every=200, 
                 save_check_dir="../checkpoint",
                 save_video_dir='../video'):
        self.env = env
        self.agent = agent
        self.n_episodes = n_episodes
        self.curr_step = 0
        self.log_every = log_every
        self.save_check_dir = save_check_dir
        self.save_video_dir = save_video_dir
        
    def train(self):
        step = 0
        # create if not exists
        self.save_check_dir.mkdir(parents=True)
        self.save_video_dir.mkdir(parents=True)
        logger = MetricLogger(self.save_check_dir)
        
        
        for e in range(self.n_episodes):
            # reset environment
            done, trunc = False, False
            state = self.env.reset()
            measure_array = []
            while (not done) and (not trunc):
                # 1. get action for state
                start = datetime.datetime.now()
                action = self.agent.perform_action(state) # 20.69 ms
                # 2. run action in environment
                next_state, reward, done, trunc, _ = self.env.step(action) # 1.70 ms
                # 3. collect experience in exp. replay buffer for Q-learning
                self.agent.store_transition(state, action, reward, next_state, done, trunc) # 0.56 ms
                # 4. Learn from collected experiences
                q, loss = self.agent.learn(step) # 39.76 ms
                # 5. Update the current state 
                state = next_state
                # 6. Update step value 
                step += 1
                measure = datetime.datetime.now() - start
                measure = measure.total_seconds() * 1000                
                # Logging
                logger.log_step(reward, loss, q)
                measure_array.append(measure)
                
            avg_measure = sum(measure_array)/len(measure_array)
            print(f"Avg. step time for measure: {avg_measure:.2f} ms")
            
            if e > 0: 
                save_video(
                    self.env.render(),
                    self.save_video_dir,
                    fps=self.env.metadata["render_fps"],
                    episode_index=e
                )
                
            logger.log_episode()
            if (not(e % self.log_every) and e > 0) or (e == self.n_episodes - 1) :
                logger.record(episode=e, epsilon=self.agent.gamma, step=step)
                
            
        
        
        