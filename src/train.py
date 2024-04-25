from agents.dqn_agent import DQNAgent
from trainer import Trainer
from gym.wrappers import FrameStack
from utils.wrappers import SkipFrame, GrayScaleObservation, ResizeObservation

import gymnasium as gym
import argparse
import torch
from pathlib import Path
import datetime

if __name__ == '__main__':
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using torch with {device}")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--environment", help='The name of environment where the agent is going to perform.', default="ALE/DemonAttack-v5")
    parser.add_argument("-l", "--log_path", help='The path where the logs are going to be stored.')
    parser.add_argument("-s", "--save_checkpoint", help='The path where the weights from the model are going to be stored.')
    parser.add_argument("-k", "--skip_frames", help='Tells the number of frames to skip and stack for the observation.', default=4)
    parser.add_argument("-b", "--batch_size", help='batch size for the experience replay learning of the DQN.', default=32)
    parser.add_argument("-f", "--discount_factor", help="Discount factor for Q-learning updates", default=0.9)
    parser.add_argument("-n", "--number_episodes", help="Number of episodes", default=40000)
    parser.add_argument("-t", "--type", help="Which type of Tranformer to use for vision", default="patch_transformer")
    parser.add_argument("--save_check_dir", help="Path to the folder where checkpoints are stored", default="../checkpoints")
    parser.add_argument("--save_video_dir", help="Path to the folder where videos of the agent playing are stored", default="../videos")
    parser.add_argument("--save_net_every", help="How many steps between saving the network", default=5e5)
    # process the arguments, store them in args
    args = parser.parse_args()
    # create the environment and add the wrappers
    env = gym.make(args.environment, render_mode='rgb_array_list')
    env = SkipFrame(env, skip=args.skip_frames)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=args.skip_frames)
    env.metadata["render_fps"] = 40
    
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n
    
    save_check_dir = Path(args.save_check_dir) / args.environment / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_video_dir = Path(args.save_video_dir) / args.environment / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    
    agent = DQNAgent(type=args.type,
                     obs_shape=obs_shape,
                     action_dim=n_actions,
                     device=device,
                     batch_size=args.batch_size,
                     gamma=args.discount_factor,
                     save_every=args.save_net_every,
                     save_net_dir=save_check_dir
                    )
    
    trainer = Trainer(env=env, 
                      agent=agent, 
                      n_episodes=args.number_episodes, 
                      log_every=2,
                      save_check_dir=save_check_dir,
                      save_video_dir=save_video_dir)
    
    trainer.train()
    
    print("Training finished!")
    
    