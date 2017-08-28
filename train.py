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

from model import ICM

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
    if icm is None:
      icm = ICM(my_env.DoomWrapper.input_channels, my_env.DoomWrapper.action_size) 
      print("init local icm")

    env = my_env.DoomWrapper()

    if rank == 0:
        configure(args.model, flush_secs=5)

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
        old_states = []
        new_states = []
        actions = []

        for step in range(args.num_steps):
            value, logit, (hx, cx) = model(
                (Variable(state.unsqueeze(0)), (hx, cx)))
            prob = F.softmax(logit)
            log_prob = F.log_softmax(logit)
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)

            action = prob.multinomial().data
            log_prob = log_prob.gather(1, Variable(action))

            new_state, reward, done, _ = env.step(action.numpy())
            done = done or episode_length >= args.max_episode_length

            frames.value += 1

            old_states.append(state)
            actions.append(action)
            new_states.append(preprocess(new_state))

            myR += reward
            state = new_state

            reward = max(min(reward, 1.0), -1.0)
            
            state = preprocess(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                if rank == 0:
                  log_value("return", myR, frames.value)
                  #log_value("intrinsic return", myIR, frames.value)
                
                print(frames.value, ": R=", myR)

                episode_length = 0
                myR, myIR = 0, 0
                state = env.reset()
                state = preprocess(state)

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

        old_states = torch.stack(old_states)
        actions = torch.cat(actions)
        new_states = torch.stack(new_states)
        intrinsic_rewards, icm_loss, inv_loss = icm(Variable(old_states), Variable(actions), Variable(new_states))

        gae = torch.zeros(1, 1)

        icm_coef = 1.0 if args.icm else 0.0

        for i in reversed(range(len(rewards))):
            R = args.gamma * R + float(rewards[i] + icm_coef * intrinsic_rewards.data.numpy()[i])
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = rewards[i] + args.gamma * \
                values[i + 1].data - values[i].data
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                log_probs[i] * Variable(gae) - 0.01 * entropies[i]

        optimizer.zero_grad()

        total_loss = (policy_loss + 0.5 * value_loss) + 10.0 * icm_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 40)

        if rank == 0:
            n = len(rewards)
            log_value("icm loss", icm_loss.data.numpy()[0] / n, frames.value)
            log_value("value loss", value_loss.data.numpy()[0] / n, frames.value)
            log_value("policy loss", policy_loss.data.numpy()[0] / n, frames.value)
            log_value("inv loss", inv_loss.numpy()[0] / n, frames.value)

        ensure_shared_grads(model, shared_model)
        optimizer.step()

