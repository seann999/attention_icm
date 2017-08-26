from vizdoom import *
import numpy as np
import gym
import scipy.misc
import matplotlib.pyplot as plt

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

class DoomWrapper:
    input_channels = 4
    action_size = 3

    def __init__(self):
        try:
            game = DoomGame()
            game.load_config("/home/sean/ViZDoom/scenarios/my_way_home.cfg")
            game.set_window_visible(False)
            game.init()
        except Exception as e:
            print(e)

        self.game = game
        self.n_step = 0
        self.last_state = None

    def seed(self, seed):
        self.game.set_seed(seed)

    def get_state(self):
        state = self.game.get_state()
        
        if not state:
            return self.last_state

        state = np.array(state.screen_buffer)
        state = rgb2gray(state)
        state = scipy.misc.imresize(state, (42, 42)) / 255.0
        #state = np.moveaxis(state, 2, 0)
        self.last_state = np.expand_dims(state, 0)

        return self.last_state

    def reset(self):
        self.game.new_episode()
        state = self.get_state()

        return np.concatenate([state]*4)

    def step(self, action):
        reward = 0
        states = []
        self.game.set_action([1 if i == action else 0 for i in range(DoomWrapper.action_size)])
         
        for _ in range(4):
          self.game.advance_action(1)
          reward += self.game.get_last_reward()
          new_state = self.get_state()
          states.append(new_state)
          
        state = np.concatenate(states)

        return state, reward, self.game.is_episode_finished(), None

class AtariWrapper:
    def __init__(self, args):
        game = gym.make(args.env_name)

    def seed(self, seed):
        pass

    def reset(self):
        pass
 
