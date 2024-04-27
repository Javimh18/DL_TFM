import torch
from torchvision import transforms as T
from pathlib import Path
import datetime

import gymnasium as gym
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace
from wrappers import SkipFrame, GrayScaleObservation, ResizeObservation

import gym_super_mario_bros
from logger import MetricLogger
from mario_agent import Mario

SKIP_FRAME = 4

if __name__ == '__main__':

    # Initializing the environment
    if gym.__version__ < '0.26':
        env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", new_step_api=True)
    else:
        env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode="rgb", apply_api_compatibility=True)

    # To make things a little bit easier, we limit the space to:
    #   0. walk right
    #   1. jump right
    # https://stackoverflow.com/questions/76509663/typeerror-joypadspace-reset-got-an-unexpected-keyword-argument-seed-when-i
    # JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)
    # env = JoypadSpace(env, [["right"], ["right", "A"]])

    # Apply Wrappers to environment
    env = SkipFrame(env, skip=SKIP_FRAME)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    if gym.__version__ < '0.26':
        env = FrameStack(env, num_stack=SKIP_FRAME, new_step_api=True)
    else:
        env = FrameStack(env, num_stack=SKIP_FRAME)

    env.reset()
    next_state, reward, done, trunc, info = env.step(action=0)
    print(f"{next_state.shape},\n {reward},\n {done},\n {info}")

    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")
    print()

    save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)

    mario = Mario(state_dim=(SKIP_FRAME, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)

    logger = MetricLogger(save_dir)

    episodes = 40000
    for e in range(episodes):

        state = env.reset()
        measure_array = []
        while True:
            start = datetime.datetime.now()
            # Run agent on the state
            action = mario.act(state)

            # Agent performs action
            next_state, reward, done, trunc, info = env.step(action)

            # Remember
            mario.cache(state, next_state, action, reward, done)

            # Learn
            q, loss = mario.learn()

            # Logging
            logger.log_step(reward, loss, q)

            # Update state
            state = next_state
            
            measure = datetime.datetime.now() - start
            measure = measure.total_seconds() * 1000

            # Check if end of game
            if done or info["flag_get"] or trunc:
                break
            measure_array.append(measure)
            
        avg_measure = sum(measure_array)/len(measure_array)
        print(f"Avg. step ({mario.curr_step}) time for measure: {avg_measure:.2f} ms")
        logger.log_episode()

        if (e % 20 == 0) or (e == episodes - 1):
            logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)