# geintra_run_python_env -d logs/ -M 32000 -g -G 1 -V -C rl_env_tfm train.py -x pow -o 1 -n 7e6 
import gymnasium as gym
from gym.utils.save_video import save_video
import torch

from agents.dqn_agent import DQNAgent
from agents.ddqn_agent import DDQNAgent
from utils.wrappers import SkipFrame, FrameExtractor, SaveOriginalObservation, NormalizeObservation

import argparse
import yaml
from pathlib import Path
import datetime
import numpy as np

SEED = 1234
FRAME_SKIP = 4
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def first_if_tuple(x):
    return x[0] if isinstance(x, tuple) else x
    
def make_env():
    env = gym.make(args.environment, render_mode='rgb_array')
    env = SaveOriginalObservation(env)
    env = FrameExtractor(env)
    env = SkipFrame(env, skip=FRAME_SKIP)
    env = gym.wrappers.ResizeObservation(env, (84,84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = NormalizeObservation(env)
    env = gym.wrappers.FrameStack(env, FRAME_SKIP)
    env.action_space.seed(SEED)
    return env

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--environment", help='The name of environment where the agent is going to perform.', default="ALE/MsPacman-v5")
    parser.add_argument("-c", "--agent_config", help="Path to the config file of the agent", default="../config/agents_config.yaml")
    parser.add_argument("-m", "--agent_model_config", help="Path to the config file of the model that the agent uses as a function approximator", default="../config/agent_nns.yaml")
    parser.add_argument("-p", "--path_to_agent_checkpoint", help="Path to the checkpoint file where that contains the NN weights", 
                        default='../checkpoints/ALE/MsPacman-v5/ddqn_swin_transformer_agent/2024-05-31T08-35-17')
    parser.add_argument("-v", "--save_video_dir", help="Path to the folder where videos of the agent playing are stored", default="../evidences/videos")
    parser.add_argument("-d", "--cuda_device", help="Cuda device to train on.", default=None)
    
    # process the arguments, store them in args
    args = parser.parse_args()
    if torch.cuda.device_count() > 1:
        if args.cuda_device is not None:
            device = torch.device(f'cuda:{args.cuda_device}')
        else:
            device = torch.device(f'cuda') if torch.cuda.is_available() else torch.device('cpu')
    else:
        device = torch.device(f'cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using torch with {device}")
    
    env = make_env()
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n
        
    # load the agent configuration and the agent model's configuration
    with open(args.agent_config, 'r') as f:
        agent_config = yaml.load(f, Loader=yaml.SafeLoader)
    with open(args.agent_model_config, 'r') as f:
        nn_config = yaml.load(f, Loader=yaml.SafeLoader)    
    
    agents_list = list(agent_config.keys())
    for i in range(len(agents_list)):
        print(f"Option {i+1}: {agents_list[i]}")
    choice = input(">Option:")
    choice = int(choice)
    while choice <= 0 or choice > len(agents_list):
        print("Incorrect option, please try again.")
        choice = input("> Option:")
        choice = int(choice)
        
    # Selected agent
    agent = agents_list[choice-1]
    print(f">>>>>>>> Selected agent: {agent}")
    
    save_video_dir = Path(args.save_video_dir) / args.environment / agent / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    
    if 'dqn' in agent:
        agent = DQNAgent(obs_shape=obs_shape,
                        action_dim=n_actions,
                        device=device,
                        save_net_dir="",
                        exp_schedule="lin",
                        prioritized_replay=False,
                        agent_config=agent_config[agent],
                        nn_config=nn_config
                        )
    elif 'ddqn' in agent:
        agent = DDQNAgent(obs_shape=obs_shape,
                        action_dim=n_actions,
                        device=device,
                        save_net_dir="",
                        prioritized_replay=False,
                        exp_schedule="lin",
                        agent_config=agent_config[agent],
                        nn_config=nn_config
                        )
        
    agent.load_weights(path_to_checkpoint=args.path_to_agent_checkpoint)
    agent.exploration_rate = 0.0
    
    curr_step = 0
    actions = []
    # PLAY WITH AGENT!!
    done, trunc = False, False
    state = env.reset()
    trun_frames = []
    original_frames = []
    total_reward = 0
    while (not done) and (not trunc):
        # 1. get action for state
        state = first_if_tuple(state).__array__()
        state = torch.tensor(state, device=device).unsqueeze(0)
        q_values = agent.net(state.float(), model='online')
        action = torch.argmax(q_values, dim=1).item()
        actions.append(action)
        # 2. run action in environment
        next_state, reward, done, trunc, info = env.step(action) # 1.70 ms
        trun_frames.append(np.array(env.frames))
        original_frames.append(env.get_original_observation())
        # 4. Learn from collected experiences
        state = next_state
        # 6. Update step value 
        curr_step += 1   
        total_reward += reward
        
    save_video(
        original_frames,
        save_video_dir,
        fps=15,
        episode_index=0
    )    
    
    print(original_frames[0].shape)
    print(trun_frames[0].shape)
    
    print(f"Trun. Frames length: {len(trun_frames)}",
          f"Original. Frames length: {len(original_frames)}",
          f"Actions length: {len(actions)}",
          f"Reward: {total_reward}")
    
            
   