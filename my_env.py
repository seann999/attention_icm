from vizdoom import *
import numpy as np
import gym
import scipy.misc
import matplotlib.pyplot as plt

class DoomWrapper:
    input_channels = 3
    action_size = 3

    def __init__(self):
        try:
            game = DoomGame()
            game.load_config("/Users/sean/git/ViZDoom/scenarios/my_way_home.cfg")
            game.set_window_visible(False)
            game.init()
        except Exception as e:
            print(e)

        self.game = game
        self.n_step = 0

    def seed(self, seed):
        self.game.set_seed(seed)

    def get_state(self):
        state = self.game.get_state()
        
        if not state:
            return None

        state = np.array(state.screen_buffer)
        state = scipy.misc.imresize(state, (42, 42)) / 255.0
        state = np.moveaxis(state, 2, 0)

        return state

    def reset(self):
        self.game.new_episode()
        state = self.get_state()

        return state

    def step(self, action):
        reward = self.game.make_action([1 if i == action else 0 for i in range(DoomWrapper.action_size)])
        state = self.get_state()

        return state, reward, self.game.is_episode_finished(), None

class AtariWrapper:
    def __init__(self, args):
        game = gym.make(args.env_name)

    def seed(self, seed):
        pass

    def reset(self):
        pass

    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])