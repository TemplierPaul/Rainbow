import procgen
import gym
from env import *
import numpy as np
import cv2
cv2.ocl.setUseOpenCL(False)
from gym import spaces
import torch 

class TorchTransposeWrapper(gym.Wrapper):
    """
    Re-order channels, from HxWxC to CxHxW.
    It is required for PyTorch convolution layers.
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = self.transpose_space(env.observation_space)

        
    @staticmethod
    def transpose_space(observation_space):
        height, width, channels = observation_space.shape
        new_shape = (channels, height, width)
        return spaces.Box(low=0, high=255, shape=new_shape, dtype=observation_space.dtype)

    @staticmethod
    def transpose_image(image):
        if len(image.shape) == 3:
            image = np.transpose(image, (2, 0, 1))
        else:
            image = np.transpose(image, (0, 3, 1, 2))
        return torch.tensor(image, dtype=torch.float32)
    
    def transpose_observation(self, obs):
        obs = self.transpose_image(obs)
        return obs
    
    def reset(self):
        return self.transpose_observation(self.env.reset())
    
    def step(self, action):
        s, r, d, i = self.env.step(action)
        s = self.transpose_observation(s)
        return s, r, d, i 

def is_procgen(name):
    return name.split("-")[0] in procgen.env.ENV_NAMES

def make_procgen_env(env_id, seed, render=False):
    try:
        env_type = env_id.split("-")[1]
        env_name = env_id.split("-")[0]
    except:
        env_type = "easy"
        env_name = env_id

    simple_graphics = False
    if env_type == "simple":
        env_type = "easy"
        simple_graphics = True

    env = procgen.gym_registration.make_env(
        env_name=env_name, 
        distribution_mode=env_type,
        rand_seed=seed,
        use_monochrome_assets= simple_graphics,
        restrict_themes=simple_graphics,
        use_backgrounds=not simple_graphics,
        render=render
        )
    # env = WarpFrame(env)
    env = TorchTransposeWrapper(env)
    return env

class ProcEnv(Env):
    def __init__(self, env_id, seed, render=False):
        # super().__init__()
        self.env_id = env_id
        self.seed = seed
        self.render = render
        self.env = None
        self.reset()
        self.training = True  # Consistent with model training mode

    def reset(self, seed=None):
        if seed is not None: self.seed = seed
        self.env = make_procgen_env(self.env_id, self.seed, self.render)
        obs = self.env.reset()
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # obs to torch tensor
        # obs = torch.from_numpy(obs)
        return obs, reward, done

    def render(self, mode='human'):
        self.env.render(mode)

    def action_space(self):
        return self.env.action_space.n


    