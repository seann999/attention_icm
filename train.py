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
import my_env

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def preprocess(state):
    return torch.from_numpy(state).float()

def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)

def train_model(rank, args, shared_model, icm, frames, optimizer=None):
    env = my_env.DoomWrapper()

    try:
        configure(args.model, flush_secs=5)
    except:
        pass

    torch.manual_seed(args.seed + rank)

    env.seed(args.seed + rank)

    model = ActorCritic(env.input_channels, env.action_size)

    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    model.train()

    state = env.reset()
    state = preprocess(state)
    #state = torch.from_numpy(state)
    done = True

    episode_length = 0

    myR, myIR = 0, 0
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

            frames.value += 1

            if done:
                log_value("return", myR, frames.value)
                
                print(frames.value, ": R=", myR, " IR=", myIR)

                episode_length = 0
                myR, myIR = 0, 0
                state = env.reset()
                state = preprocess(state)
            else:
                intrinsic_reward, icm_l, _ = icm(state, action, preprocess(old_state))
                icm_loss += icm_l

                intrinsic_reward = intrinsic_reward.numpy()[0]

                myR += reward
                myIR += intrinsic_reward

                state = old_state
                reward += intrinsic_reward
                reward = max(min(reward, 1), -1)

                state = preprocess(state)
                #state = torch.from_numpy(state)
                values.append(value)
                log_probs.append(log_prob)
                rewards.append(reward)

            if frames.value % args.save_frames == 0:
                save_checkpoint({
                    'frames': frames.value,
                    'a3c': shared_model.state_dict(),
                    'icm': icm.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, '{}/checkpoint-{}.pth'.format(args.model, frames.value))

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

        total_loss = 0.1 * (policy_loss + 0.5 * value_loss) + 10.0 * icm_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 40)

        log_value("icm loss", icm_loss.data.numpy()[0], frames.value)

        ensure_shared_grads(model, shared_model)
        optimizer.step()
