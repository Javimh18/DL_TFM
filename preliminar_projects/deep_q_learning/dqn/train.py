from dqn_agent import Agent
from trainer import Trainer
from gym.wrappers import FrameStack
from wrappers import SkipFrame, GrayScaleObservation, ResizeObservation
import gymnasium as gym
import argparse
import torch

if __name__ == '__main__':
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--environment", help='The name of environment where the agent is going to perform.', default="ALE/MarioBros-v5")
    parser.add_argument("-l", "--log_path", help='The path where the logs are going to be stored.')
    parser.add_argument("-s", "--save_checkpoint", help='The path where the weights from the model are going to be stored.')
    parser.add_argument("-k", "--skip_frames", help='Tells the number of frames to skip and stack for the observation.', default=4)
    parser.add_argument("-b", "--batch_size", help='batch size for the experience replay learning of the DQN.', default=32)
    parser.add_argument("-f", "--discount_factor", help="Discount factor for Q-learning updates", default=0.9)
    parser.add_argument("-n", "--number_episodes", help="Number of episodes", default=4000)
    # process the arguments, store them in args
    args = parser.parse_args()
    # create the environment and add the wrappers
    env = gym.make(args.environment)
    env = SkipFrame(env, skip=args.skip_frames)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=args.skip_frames)
    
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n
    
    agent = Agent(state_dim=obs_shape,
                  action_dim=n_actions,
                  device=device,
                  batch_size=args.batch_size,
                  gamma=args.discount_factor
                  )
    
    trainer = Trainer(env, agent, args.number_episodes)
    
    trainer.train()
    
    print("Training finished!")
    
    