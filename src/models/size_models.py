from torchsummary import summary
import torch

import gymnasium as gym
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv
)

from patch_transformer import PatchTransformer
from vit import ViT
from cnn import CNN
from models.swin_transformer import SwinTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)

        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)

        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.action_space.seed(seed)

        return env
    
    return thunk

envs = gym.vector.SyncVectorEnv(
        [make_env("BreakoutNoFrameskip-v4", 1234 + i, i, True, "lel") for i in range(1)]
    )
obs_shape = envs.single_observation_space.shape
n_actions = envs.single_action_space.n
print(obs_shape[0])
"""
print(">>>>>>>>> Conv NN")
q_network = CNN(n_actions=n_actions).to(device)
summary(q_network, obs_shape)
"""
print(">>>>>>>>> Patch Transformer:")
q_network = PatchTransformer(n_layers=2, n_actions=envs.single_action_space.n, patch_size=4, \
    fc_dim=64, embed_dim=128, head_dim=256, attn_heads=[4,8], dropouts=[0.3, 0.3], input_shape=(1,)+obs_shape).to(device)
summary(q_network, obs_shape)
print(">>>>>>>>> Vision Transformer:")
q_network = ViT(img_size=obs_shape, patch_size=(3,3), embed_dim=144, n_heads=6, n_layers=7, n_actions=envs.single_action_space.n).to(device)
summary(q_network, obs_shape)
print(">>>>>>>> SWIN Transformer")
q_network = SwinTransformer(img_size=84, patch_size=3, in_chans=4, num_classes=envs.single_action_space.n, embed_dim=96,
                            depths=[2,3,2], num_heads=[3,3,6], window_size=7).to(device)
summary(q_network, (4,84,84))

