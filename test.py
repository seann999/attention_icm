import math
import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
from model import ActorCritic
from torch.autograd import Variable
from torchvision import datasets, transforms
from tensorboard_logger import configure, log_value
import cv2

import gym
import numpy as np
import scipy.misc
import my_env

def preprocess(state):
    return torch.from_numpy(state).float()

def test_model(rank, args, shared_model, frames):
    if rank == -1:
        cv2.namedWindow("test", cv2.WINDOW_NORMAL)

    env = my_env.DoomWrapper(4)
    #env = my_env.AtariWrapper(args)

    torch.manual_seed(args.seed + rank)

    env.seed(args.seed + rank)

    model = ActorCritic(env.input_channels, env.action_size)
    model.eval()

    state = env.reset()
    state = preprocess(state)

    #state = torch.from_numpy(state)
    done = True

    episode_length = 0

    myR, myIR = 0, 0
    while True:
        episode_length += 1

        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())
            cx = Variable(torch.zeros(1, 256))
            hx = Variable(torch.zeros(1, 256))
        else:
            cx = Variable(cx.data)
            hx = Variable(hx.data)

        value, logit, (hx, cx) = model(
            (Variable(state.unsqueeze(0)), (hx, cx)))
        prob = F.softmax(logit)

        action = np.argmax(prob.data.numpy())#prob.multinomial().data

        new_state, reward, done, _ = env.step(action)
        done = done or episode_length >= args.max_episode_length

        raw_new_state = new_state
        new_state = preprocess(new_state)

        if rank == -1:
            cv2.imshow("test", raw_new_state[0])
            cv2.waitKey(1)

        myR += reward

        state = new_state

        reward = max(min(reward, 1.0), -1.0)

        if done:
            log_value("test return", myR, frames.value)
            
            print(frames.value, ": R=", myR)

            episode_length = 0
            myR, myIR = 0, 0
            state = env.reset()
            state = preprocess(state)  