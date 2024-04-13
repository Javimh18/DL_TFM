import torch
from torchvision import transforms as T
import gym
from gym.spaces import Box
import numpy as np

###########################################################################################
############################ SKIP FRAME WRAPPER ###########################################
###########################################################################################
# Custom wrapper that implements the step function. Since between consecutive frames there
# is not much variation, we can skip n-intermidiate frames w/o losing too much information.
# The n-th frame aggregates rewards accumulated over each skipped frame.

class SkipFrame(gym.Wrapper):
    def __init__(self, env: gym.Env, skip):
        """Returns only the `skip`-th frame."""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            if done or trunk:
                break
        
        return obs, total_reward, done, trunk, info
    
###############################################################################################
################################ GRAY SCALE OBSERVATION #######################################
###############################################################################################
# Since colors are not relevant information for the agent when playing Mario, we can reduce
# the environment's state size from [3, 240, 256] to [1, 240, 256]. 

class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        # since we are wrapping the observation that an environment provides,
        # we must update the observation space to match the new wrapped env.
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor for pytorch model
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation
    
    def observation(self, observation):
        observation = self.permute_orientation(observation)
        # since we have updated the observation space attribute, we must 
        # do so with the pixels from the env, casting them from RGB to grayscale
        transform = T.Grayscale()
        observation = transform(observation)
        return observation
    

###############################################################################################
################################### RESIZE OBSERVATION ########################################
###############################################################################################
# Downsample the observations from [1, 240, 256] to [1, 84, 84] to ease the input to a feature 
# extractor (i.e. a CNN).
    
class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, shape):
        super().__init__(env)
        # If shape is just a number it creates a tuple with that same number (i.e. 84 -> (84,84))
        # if is a list creates a "tuple" object with the given shape
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)
        
        obs_shape = self.shape # + self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        # Use torch transforms to resize the observation 
        # to the wanted resize resolution
        transforms = T.Compose(
            [T.Resize(self.shape, antialias=True),
             T.Normalize(0,255)]
        )

        # apply the transformation
        observation = transforms(observation).squeeze(0)
        return observation