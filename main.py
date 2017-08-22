from __future__ import print_function

import argparse
import os
import sys

import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from model import ActorCritic
from train import train
import my_optim
from model import ICM
import glob
import my_env

# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--tau', type=float, default=1.00, metavar='T',
                    help='parameter for GAE (default: 1.00)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--num-processes', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 4)')
parser.add_argument('--num-steps', type=int, default=20, metavar='NS',
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=10000, metavar='M',
                    help='maximum length of an episode (default: 10000)')
parser.add_argument('--env-name', default='PongDeterministic-v4', metavar='ENV',
                    help='environment to train on (default: PongDeterministic-v4)')
parser.add_argument('--no-shared', default=False, metavar='O',
                    help='use an optimizer without shared momentum.')
parser.add_argument('--model', type=str)
parser.add_argument('--save_frames', type=int, default=1000,
                    help='save every n frames')

def atari():
    return gym.make(args.env_name)

def doom():
    return my_env.DoomWrapper()

if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'  
  
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    #env = create_atari_env(args.env_name)
    shared_model = ActorCritic(1, 5)
    shared_model.share_memory()

    if args.no_shared:
        optimizer = None
    else:
        optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=args.lr)
        optimizer.share_memory()

    icm = ICM(1, 5)
    frames = mp.Value('i', 0)

    ckpt_path = None
    paths = glob.glob(os.path.join(args.model, "checkpoint-*"))
    if len(paths) > 0:
        paths.sort()
        ckpt_path = paths[-1]

    if ckpt_path and os.path.isfile(ckpt_path):
        print("=> loading checkpoint '{}'".format(ckpt_path))
        checkpoint = torch.load(ckpt_path)
        frames.value = checkpoint['frames']
        shared_model.load_state_dict(checkpoint['a3c'])
        icm.load_state_dict(checkpoint['icm'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (frame {})".format(ckpt_path, checkpoint['frames']))
    else:
        print("=> no checkpoint found at '{}'".format(args.model))

    processes = []

    #p = mp.Process(target=test, args=(args.num_processes, args, shared_model))
    #p.start()
    #processes.append(p)
    
    #train(0, args, shared_model, optimizer)

    for rank in range(0, args.num_processes):
        p = mp.Process(target=train, args=(rank, args, shared_model, icm, frames, doom(), optimizer))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
