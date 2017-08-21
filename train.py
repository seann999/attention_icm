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

import gym
import numpy as np
import scipy.misc

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def preprocess(state):
    gray = rgb2gray(state)
    gray = scipy.misc.imresize(gray, (42, 42)) / 255.0
    return torch.from_numpy(np.expand_dims(gray, 0)).float()

def train(rank, args, shared_model, optimizer=None, icm=None):
    if rank == 0:
        configure(args.model, flush_secs=5)

    torch.manual_seed(args.seed + rank)

    env = gym.make(args.env_name)
    env.seed(args.seed + rank)

    model = ActorCritic(1, 5)

    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    model.train()

    state = env.reset()
    state = preprocess(state)
    #state = torch.from_numpy(state)
    done = True

    episode_length = 0
    episodes_done = 0
    myR = 0
    while True:
        episode_length += 1
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        if done:
            cx = Variable(torch.zeros(1, 256))
            hx = Variable(torch.zeros(1, 256))
        else:
            cx = Variable(cx.data)
            hx = Variable(hx.data)

        values = []
        log_probs = []
        rewards = []
        entropies = []
        icm_loss = 0

        for step in range(args.num_steps):
            value, logit, (hx, cx) = model(
                (Variable(state.unsqueeze(0)), (hx, cx)))
            prob = F.softmax(logit)
            log_prob = F.log_softmax(logit)
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)

            action = prob.multinomial().data
            log_prob = log_prob.gather(1, Variable(action))

            old_state, reward, done, _ = env.step(action.numpy())
            done = done or episode_length >= args.max_episode_length

            _, icm_l, _ = icm(state, action, preprocess(old_state))
            icm_loss += icm_l

            state = old_state
            reward = max(min(reward, 1), -1)
            myR += reward

            if done:
                if rank == 0:
                    log_value("return", myR, episodes_done)
                
                print(episodes_done, ": R=", myR)
                episodes_done += 1
                episode_length = 0
                myR = 0
                state = env.reset()

            state = preprocess(state)
            #state = torch.from_numpy(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((Variable(state.unsqueeze(0)), (hx, cx)))
            R = value.data

        values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = rewards[i] + args.gamma * \
                values[i + 1].data - values[i].data
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                log_probs[i] * Variable(gae) - 0.01 * entropies[i]

        optimizer.zero_grad()

        total_loss = (policy_loss + 0.5 * value_loss) + icm_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 40)

        ensure_shared_grads(model, shared_model)
        optimizer.step()
