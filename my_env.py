from vizdoom import *
import numpy as np
import gym
import scipy.misc
import matplotlib.pyplot as plt

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

class DoomWrapper:
    input_channels = 4
    action_size = 4

    def __init__(self, action_repeat):
        try:
            game = DoomGame()
            game.load_config("/home/sean/ViZDoom/scenarios/my_way_home.cfg")
            game.set_window_visible(False)
            game.init()
        except Exception as e:
            print(e)

        self.game = game

        self.action_repeat = action_repeat

        self.buffer = []
        self.obs_buffer = []

    def seed(self, seed):
        self.game.set_seed(seed)

    def preprocess(obs):
        obs = np.array(obs)
        self.obs_buffer.append(obs)
        max_frame = np.max(np.stack(self.obs_buffer), 0)
        obs = scipy.misc.imresize(max_frame, (42, 42))
        obs = rgb2gray(obs) / 255.0

        return obs

    def reset(self):
        self.obs_buffer = []
        self.game.new_episode()
        obs = preprocess(self.game.get_state())
        self.buffer = []
        for _ in range(DoomWrapper.input_channels):
            self.buffer.append(np.zeros_like(obs))

        return np.concatenate(states)

    def step(self, action):
        self.game.set_action([1 if i == action else 0 for i in range(DoomWrapper.action_size-1)])

        total_reward = 0

        for _ in range(self.action_repeat):
            self.game.advance_action(1)
            reward = self.game.get_last_reward()
            reward = max(reward, 0)
            obs = preprocess(self.game.get_state())
            total_reward += reward
        self.buffer.append(obs)
        
        states = []

        past = 4
        for i in range(past):
            states.append(self.buffer[max(-1-i, -len(self.buffer))])

        state = np.concatenate(states)

        return state, total_reward, self.game.is_episode_finished(), None

class AtariWrapper:
    input_channels = 4
    action_size = 6

    def __init__(self, args):
        self.game = gym.make(args.env_name)
        self.last_state = None
        self.past_states = []

    def seed(self, seed):
        self.game.seed(seed)

    def process(self, state):
        state = scipy.misc.imresize(state, (42, 42))
        state = rgb2gray(state) / 255.0
        self.last_state = np.expand_dims(state, 0)

        return self.last_state

    def reset(self):
        self.past_states = []
        state = self.process(self.game.reset())
        self.past_states.append(state)

        return np.concatenate([state]*4)

    def step(self, action):
        new_state, reward, done, _ = self.game.step(action)
        new_state = self.process(new_state)
        self.past_states.append(new_state)
        
        states = []

        past = 4
        for i in range(past):
            states.append(self.past_states[max(-1-i, -len(self.past_states))])

        state = np.concatenate(states)

        return state, reward, done, None
 
