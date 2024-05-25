from agents.dqn_agent import DQNAgent
from agents.ddqn_agent import DDQNAgent
from trainer import Trainer
from gym.wrappers import FrameStack, RecordEpisodeStatistics
from utils.wrappers import SkipFrame, GrayScaleObservation, ResizeObservation
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv
)

import gymnasium as gym
import argparse
import torch
from pathlib import Path
import datetime
import yaml

SEED = 1234    

# create the environment, add the wrappers and extract important parameters
def make_env():
    env = gym.make(args.environment, render_mode='rgb_array_list')
    env = RecordEpisodeStatistics(env)
    env = SkipFrame(env, skip=args.skip_frames)
    env = EpisodicLifeEnv(env)
    env = ClipRewardEnv(env)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=args.skip_frames)
    env.metadata["render_fps"] = 30
    env.action_space.seed(SEED)
    return env

if __name__ == '__main__':

    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using torch with {device}")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--environment", help='The name of environment where the agent is going to perform.', default="ALE/MsPacman-v5")
    parser.add_argument("-k", "--skip_frames", help='Tells the number of frames to skip and stack for the observation.', default=4)
    parser.add_argument("-n", "--number_steps", help="Number of steps", default=1e7)
    # parser.add_argument("-a", "--agent", help="Type of agent to train (dqn/ddqn)", default="dqn")
    parser.add_argument("-s", "--save_check_dir", help="Path to the folder where checkpoints are stored", default="../checkpoints")
    parser.add_argument("-v", "--save_video_dir", help="Path to the folder where videos of the agent playing are stored", default="../videos")
    parser.add_argument("-l", "--log_every",  help="How many episodes between printing logger statistics", default=50, type=int)
    parser.add_argument("-c", "--agent_config", help="Path to the config file of the agent", default="../config/agents_config.yaml")
    parser.add_argument("-m", "--agent_model_config", help="Path to the config file of the model that the agent uses as a function approximator", default="../config/agent_nns.yaml")
    # process the arguments, store them in args
    args = parser.parse_args()
    
    env = make_env()
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n
    
    # load the agent configuration
    with open(args.agent_config, 'r') as f:
        agent_config = yaml.load(f, Loader=yaml.SafeLoader)
    
    # load the agent model's configuration
    with open(args.agent_model_config, 'r') as f:
        nn_config = yaml.load(f, Loader=yaml.SafeLoader)
    
    print("Select among the different agent configs:")
    agents_list = list(agent_config.keys())
    for i in range(len(agents_list)):
        print(f"Option {i+1}: {agents_list[i]}")
    
    choice = input("> Option:")
    choice = int(choice)
    while choice <= 0 or choice > len(agents_list):
        print("Incorrect option, please try again.")
        choice = input("> Option:")
        choice = int(choice)
    
    
    # Selected agent
    agent = agents_list[choice-1]
    print(f">>>>>>>> Selected agent: {agent}")
    
    # initializing key directories for metrics and evidences from the code
    save_check_dir = Path(args.save_check_dir) / args.environment / agent / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_video_dir = Path(args.save_video_dir) / args.environment / agent / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    
    if 'dqn' in agent:
        agent = DQNAgent(obs_shape=obs_shape,
                        action_dim=n_actions,
                        device=device,
                        save_net_dir=save_check_dir,
                        agent_config=agent_config[agent],
                        nn_config=nn_config
                        )
    elif 'ddqn' in agent:
        agent = DDQNAgent(obs_shape=obs_shape,
                        action_dim=n_actions,
                        device=device,
                        save_net_dir=save_check_dir,
                        agent_config=agent_config[agent],
                        nn_config=nn_config
                        )
    else:
        print("WARNING: Type of agent specified not recognized. Exiting...")
        exit()
    
    # load agent weights  
    # agent.load_weights('/home/javier.munozh/dev/DL_TFM/checkpoints/ALE/DemonAttack-v5/2024-04-28T11-30-13/_net_20_0.9_0.00025.chkpt', True)
    
    # declare the training class and start training the agent
    trainer = Trainer(env=env, 
                      agent=agent, 
                      n_steps=args.number_steps, 
                      log_every=args.log_every,
                      save_check_dir=save_check_dir,
                      save_video_dir=save_video_dir)
    trainer.train()
    env.close()
    
    print("Training finished!")
    
    