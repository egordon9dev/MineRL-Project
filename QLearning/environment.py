import gym
import minerl
from gym import spaces
import numpy as np

treechop_env = gym.make("MineRLNavigateDense-v0")

class MyEnv(gym.Env):
    """Custom Neurocar Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(MyEnv, self).__init__()
        self.action_space = spaces.Discrete(3)
        self.observation_space = treechop_env.observation_space

    def step(self, a_idx):
        a = treechop_env.action_space.noop()
        if a_idx == 0:
            # forward
            a["forward"] = 1
            a["jump"] = 1
            a["attack"] = 1
        elif a_idx == 1:
            # left
            a["camera"] = [0, -1]
            a["attack"] = 1
        elif a_idx == 2:
            # right
            a["camera"] = [0, 1]
            a["attack"] = 1
        return treechop_env.step(a) 

    def reset(self):
        return treechop_env.reset()

    def render(self, mode='human'):
        return treechop_env.render(mode)

    def close (self):
        treechop_env.close()
