import moviepy.editor as mpy
import torch
import argparse
import gymnasium as gym
import yaml
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import datetime
import glob

from utils.wrappers import SkipFrame, FrameExtractor, SaveOriginalObservation
from agents.ddqn_agent import DDQNAgent

VIT_CHOICE = 3
SEED = 1234
FRAME_SKIP = 4
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
    
# process the arguments, store them in args
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--environment", help='The name of environment where the agent is going to perform.', default="ALE/MsPacman-v5")
parser.add_argument("-c", "--agent_config", help="Path to the config file of the agent", default="../config/agents_config.yaml")
parser.add_argument("-m", "--agent_model_config", help="Path to the config file of the model that the agent uses as a function approximator", default="../config/agent_nns.yaml")
parser.add_argument("-p", "--path_to_agent_checkpoint", help="Path to the checkpoint file where that contains the NN weights", 
                    default='../checkpoints/ALE/MsPacman-v5/dqn_vit_agent/2024-06-01T09-00-00')
parser.add_argument("-v", "--save_video_dir", help="Path to the folder where videos of the agent playing are stored", default="../evidences/attention_maps")
parser.add_argument("-d", "--cuda_device", help="Cuda device to run inference on.", default=None)
args = parser.parse_args()
        
# load the agent configuration and the agent model's configuration
with open(args.agent_config, 'r') as f:
    agent_config = yaml.load(f, Loader=yaml.SafeLoader)
with open(args.agent_model_config, 'r') as f:
    nn_config = yaml.load(f, Loader=yaml.SafeLoader)    
    
def first_if_tuple(x):
    return x[0] if isinstance(x, tuple) else x

def make_env():
    env = gym.make(args.environment, render_mode='rgb_array')
    env = SaveOriginalObservation(env)
    env = FrameExtractor(env)
    env = SkipFrame(env, skip=FRAME_SKIP)
    env = gym.wrappers.ResizeObservation(env, (84,84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, FRAME_SKIP)
    env.action_space.seed(SEED)
    return env

# Function to find the original observation shape
def find_original_observation_shape(env):
    while hasattr(env, 'env'):
        env = env.env
    return env.observation_space.shape

def get_device(args):
    if torch.cuda.device_count() > 1:
        if args.cuda_device is not None:
            device = torch.device(f'cuda:{args.cuda_device}')
        else:
            device = torch.device(f'cuda') if torch.cuda.is_available() else torch.device('cpu')
    else:
        device = torch.device(f'cuda') if torch.cuda.is_available() else torch.device('cpu')
        
    return device

def interpolate_attn_map(attention_map, scale_factor, final_shape):
    attention_map = torch.nn.functional.interpolate(attention_map,
                                                    scale_factor=scale_factor,
                                                    mode='nearest')
    attention_map = torch.nn.functional.interpolate(attention_map,
                                                    size=(final_shape[0], final_shape[1]),
                                                    mode='nearest')
    attention_map = attention_map.squeeze().detach().cpu().numpy()
    return attention_map

def frames_to_video(input_folder, output_video, fps=30):
    # Get a list of all files in the directory, sorted by name
    files = sorted([f for f in os.listdir(input_folder) if f.endswith('.png') or f.endswith('.jpg')], 
                   key=lambda x: int(x.split('.')[0]))
    
    if not files:
        print("No frames found in the specified directory.")
        return

    # Load images
    image_sequence = [mpy.ImageClip(os.path.join(input_folder, file)).set_duration(1/fps) for file in files]

    # Concatenate the image sequence into a video
    video = mpy.concatenate_videoclips(image_sequence, method="compose")
    codec = 'libx264' if output_video.endswith('.mp4') else 'mpeg4'
    video.write_videofile(output_video, fps=fps, codec=codec)
    video.close()
    
    print(f"Video saved as {output_video}")
    
def remove_png_files(directory):
    # Pattern to match .png files in the specified directory
    pattern = os.path.join(directory, '*.png')
    
    # Find all files matching the pattern
    png_files = glob.glob(pattern)
    
    # Iterate over the list of files and remove each one
    for file_path in png_files:
        try:
            os.remove(file_path)
            print(f"Removed file: {file_path}")
        except Exception as e:
            print(f"Error removing file {file_path}: {e}")
            
    print("All files removed!")


if __name__ == '__main__':
    
    # TODO: WATCH OUT!!! RESULTS MAY VARY DEPENDING ON THE DEVICE YOU TRAINED YOUR MODEL ON... SELECT "cuda" BY DEFAULT
    device = get_device(args)
    print(f"INFO: Using torch with {device}")
    
    env = make_env()
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n
    original_shape = find_original_observation_shape(env) # Get the original observation shape
    
    choice = VIT_CHOICE # preselected choice for DDQN ViT agent
    agents_list = list(agent_config.keys())
    agent_conf_name = agents_list[choice-1]
    
    agent = DDQNAgent(obs_shape=obs_shape,
                        action_dim=n_actions,
                        device=device,
                        save_net_dir="",
                        exp_schedule="lin",
                        prioritized_replay=False,
                        agent_config=agent_config[agent_conf_name],
                        nn_config=nn_config
                        )
    agent.load_weights(args.path_to_agent_checkpoint)
    
    # initialize and create the directory where results are stored
    save_video_dir = Path(args.save_video_dir) / args.environment / agent_conf_name / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    os.makedirs(save_video_dir)
    
    # useful variables to rebuild the attention mnap
    patch_size = agent.net.online.patch_embed.patch_size
    dimension = int(obs_shape[2]/patch_size)
    
    curr_step = 0; total_reward=0
    actions = []
    trunc_frames = []
    original_frames = []
    attn_maps = []
    q_values = []
    
    done, trunc = False, False
    state = env.reset()
    while (not done) and (not trunc):
        # 1. get action for state
        action, q_v = agent.perform_action(state, t=-1, exploit=True)
        actions.append(action)
        q_values.append(q_v.squeeze().detach().cpu().numpy())
        # 2. get the attention maps
        state = first_if_tuple(state).__array__()
        state = torch.tensor(state, device=device).unsqueeze(0)
        vit = agent.net.online
        attn = vit.get_last_selfattention(state.float())
        cls_attn = attn[0, :, 0, 1:]
        cls_attn = cls_attn.reshape(-1, dimension, dimension).mean(dim=0, keepdim=True).unsqueeze(0)
        # 3. Interpolate the attention maps to original frame size
        cls_attn_int = interpolate_attn_map(attention_map=cls_attn, 
                                            scale_factor=patch_size,
                                            final_shape=original_shape)
        attn_maps.append(cls_attn_int)
        original_frames.append(env.get_original_observation()) # TODO: review; before or after
        next_state, reward, done, trunc, info = env.step(action)
        trunc_frames.append(np.array(env.frames))
        # 5. update current observation
        state = next_state
        # 6. Update step value
        curr_step += 1
        total_reward += reward
        
    env.close()
        
    print(f"INFO: RUN with REWARD: {total_reward}")
        
    # assert that the length between the lists is the same
    assert len(trunc_frames) == len(actions) == len(original_frames) == len(attn_maps) == len(q_values), \
        f"Lengths do not match:" \
        f"Len trunc_frames: {len(trunc_frames)}" \
        f"Len actions: {len(actions)}" \
        f"Len original frames: {len(original_frames)}" \
        f"Len attention maps: {len(attn_maps)}" \
        f"Len q values: {len(q_values)}"
        
    # extract and plot the explainability features
    frame_count = 0
    action_names = env.unwrapped.get_action_meanings()
    action_dict = {}
    for i, a in enumerate(action_names):
        action_dict[i] = a
    
    for orig_frames, attn_map, a, q_s in zip(original_frames, attn_maps, actions, q_values):
        fig, ax = plt.subplots(1, 3, figsize=(24,10))
        # subplot the original frame
        _ = ax[0].imshow(orig_frames)
        ax[0].set_title("Original Frame")
        # subplot the attention map with the color-bar
        attn_map = ax[1].imshow(attn_map)
        ax[1].set_title("Attention map")
        cbar = fig.colorbar(attn_map, ax=ax[1], cmap='plasma')
        # plot the histogram with the q values
        qs_hist = ax[2].bar(action_names, q_s, color='blue')
        plt.xticks(rotation=75)
        ax[2].set_ylim(-10,100)
        ax[2].set_title(f"Q action values with action {action_dict[a]} selected")
        plt.savefig(os.path.join(save_video_dir,
                                 f"{frame_count}.png"))
        plt.close()
        frame_count += 1
    
    output_video_path = os.path.join(save_video_dir, 'attention_viz.mp4')
    frames_to_video(save_video_dir, output_video_path, fps=8)
    # Remove .png files in the specified directory
    remove_png_files(save_video_dir)
        
    
    