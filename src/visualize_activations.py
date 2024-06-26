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
from tqdm import tqdm

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from utils.wrappers import SkipFrame, FrameExtractor, SaveOriginalObservation
from agents.ddqn_agent import DDQNAgent
from models.swin_transformer import SwinTransformer
from models.vit import ViT

VIT_CHOICE = 3
SEED = 1234
FRAME_SKIP = 4
VMIN = 0
VMAX = 1
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
    
# process the arguments, store them in args
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--environment", help='The name of environment where the agent is going to perform.', default="ALE/MsPacman-v5")
parser.add_argument("-c", "--agent_config", help="Path to the config file of the agent", default="../config/agents_config.yaml")
parser.add_argument("-m", "--agent_model_config", help="Path to the config file of the model that the agent uses as a function approximator", default="../config/agent_nns.yaml")
parser.add_argument("-p", "--path_to_agent_checkpoint", help="Path to the checkpoint file where that contains the NN weights", 
                    default='../checkpoints/ALE/MsPacman-v5/dqn_swin_transformer_agent/2024-05-31T08-35-17')
parser.add_argument("-v", "--save_video_dir", help="Path to the folder where videos of the agent playing are stored", default="../evidences/activation_maps")
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

def reshape_transform_vit(tensor: torch.Tensor, height:int=12, width:int=12):
    tensor = tensor[:,1:,:]
    B, N, E = tensor.shape
    
    assert not(N%height) and not(N%width), "VIT transform: The number of patches in the input tensor must be divisible by the H, W parameters."
    tensor = tensor.reshape(B, height, width, E)
    tensor = tensor.permute(0,3,1,2)
    
    return tensor

def reshape_transform_swin(tensor: torch.Tensor, height:int=7, width:int=7):
    B, N, E = tensor.shape
    
    assert not(N%height) and not(N%width), "SWIN Transform: The number of patches in the input tensor must be divisible by the H, W parameters."
    tensor = tensor.reshape(B, height, width, E)
    tensor = tensor.permute(0,3,1,2)
    
    return tensor


def interpolate_activations(activation_map, final_shape):
    attention_map = torch.nn.functional.interpolate(activation_map,
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
    
    agents_list = list(agent_config.keys())
    for i in range(len(agents_list)):
        print(f"Option {i+1}: {agents_list[i]}")
    choice = input(">Option:")
    choice = int(choice)
    while choice <= 0 or choice > len(agents_list):
        print("Incorrect option, please try again.")
        choice = input("> Option:")
        choice = int(choice)
    
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
    
    # select the agent's model
    model = agent.net.online
    # Construct the CAM object once, and then re-use it on many images:
    if type(model) == SwinTransformer:
        target_layer = model.layers[-1].blocks[-1].norm1
        cam = GradCAM(model=model, target_layers=[target_layer], reshape_transform=reshape_transform_swin)
    elif type(model) == ViT:
        target_layer = model.blocks[-1].norm1
        cam = GradCAM(model=model, target_layers=[target_layer], reshape_transform=reshape_transform_vit)
    else:
        print(f"Type of net {type(model)} is not allowed. Exiting...")
        exit()
        
    curr_step = 0; total_reward=0
    actions = []
    trunc_frames = []
    original_frames = []
    activation_maps = []
    q_values = []
    
    done, trunc = False, False
    state = env.reset()
    while (not done) and (not trunc):
        # 1. get action for state
        action, q_v = agent.perform_action(state, t=-1, exploit=True)
        actions.append(action)
        q_values.append(q_v.squeeze().detach().cpu().numpy())
        # 2. get the activation maps
        state = first_if_tuple(state).__array__()
        state = torch.tensor(np.array([state]), device=device)
        action_target = ClassifierOutputTarget(action)
        gray_scale_cam = cam(input_tensor=state, targets=[action_target])
        gray_scale_cam = torch.tensor(gray_scale_cam[0, :]).unsqueeze(0).unsqueeze(0) # appropiate dims for interpolation
        activ_map = interpolate_activations(gray_scale_cam,
                                             final_shape=original_shape)
        # 3. get the eseential values for plotting
        activation_maps.append(activ_map)
        trunc_frames.append(np.array(env.frames))
        original_frames.append(env.get_original_observation())
        # 4. perform step
        next_state, reward, done, trunc, info = env.step(action)
        # 5. update current observation
        state = next_state
        # 6. Update step value
        curr_step += 1
        total_reward += reward
    
    env.close()
    
    print(f"INFO: RUN with REWARD: {total_reward}")
        
    # assert that the length between the lists is the same
    assert len(trunc_frames) == len(actions) == len(original_frames) == len(activation_maps) == len(q_values), \
        f"Lengths do not match:" \
        f"Len trunc_frames: {len(trunc_frames)}" \
        f"Len actions: {len(actions)}" \
        f"Len original frames: {len(original_frames)}" \
        f"Len attention maps: {len(activation_maps)}" \
        f"Len q values: {len(q_values)}"
    
    # extract and plot the explainability features
    frame_count = 0
    action_names = env.unwrapped.get_action_meanings()
    action_dict = {}
    for i, a in enumerate(action_names):
        action_dict[i] = a
    
    for orig_frames, activ_map, a, q_s in tqdm(zip(original_frames, activation_maps, actions, q_values)):
        fig, ax = plt.subplots(1, 3, figsize=(24,10))
        # subplot the original frame
        _ = ax[0].imshow(orig_frames)
        ax[0].set_title("Original Frame")
        # subplot the attention map with the color-bar
        _ = ax[1].imshow(orig_frames)
        ax[1].set_title("Activation map")
        plot_activation_map = ax[1].imshow(activ_map, cmap='plasma', alpha=0.65, aspect='auto', vmin=VMIN, vmax=VMAX)
        cbar = fig.colorbar(plot_activation_map, ax=ax[1], cmap='plasma')
        # plot the histogram with the q values
        qs_hist = ax[2].bar(action_names, q_s, color='blue')
        plt.xticks(rotation=75)
        ax[2].set_ylim(-10,100)
        ax[2].set_title(f"Q action values with action {action_dict[a]} selected")
        plt.savefig(os.path.join(save_video_dir,
                                 f"{frame_count}.png"))
        plt.close()
        frame_count += 1
    
    output_video_path = os.path.join(save_video_dir, 'activation_viz.mp4')
    frames_to_video(save_video_dir, output_video_path, fps=4)
    # Remove .png files in the specified directory
    remove_png_files(save_video_dir)
    
