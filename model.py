import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class ActorCritic(torch.nn.Module):

    def __init__(self, num_inputs, action_space):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.lstm = nn.LSTMCell(32 * 3 * 3, 256)

        num_outputs = action_space
        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, num_outputs)

        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()

    def forward(self, inputs):
        #print("a")
        inputs, (hx, cx) = inputs
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        #print("b")
        x = x.view(-1, 32 * 3 * 3)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        return self.critic_linear(x), self.actor_linear(x), (hx, cx)

rep_size = 32 * 3 * 3

class ICM(torch.nn.Module):

    def __init__(self, num_inputs, action_space):
        super(ICM, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1),
            torch.nn.ELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            torch.nn.ELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            torch.nn.ELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            torch.nn.ELU()   
        )

        self.num_outputs  = action_space#.n

        self.act_pred = nn.Sequential(
            nn.Linear(32 * 3 * 3 * 2, 256),
            torch.nn.ELU(),
            nn.Linear(256, self.num_outputs )
        )

        self.state_pred = nn.Sequential(
            nn.Linear(32 * 3 * 3 + self.num_outputs , 256),
            torch.nn.ELU(),
            nn.Linear(256, 32 * 3 * 3)
        )

        self.train()

    def inverse_model(self, rep_old, rep_new):
        x = torch.cat([rep_old, rep_new], 1)
        logits = self.act_pred(x)
        return logits

    def forward_model(self, rep_old, act):
        x = torch.cat([rep_old, act], 1)
        pred = self.state_pred(x)
        return pred

    def forward(self, state_old, act, state_new):
        state_old = state_old.unsqueeze(0)
        state_new = state_new.unsqueeze(0)

        beta = 0.2

        rep_old = self.encoder(state_old).view(-1, rep_size)
        rep_new = self.encoder(state_new).view(-1, rep_size)

        act_onehot = torch.FloatTensor(act.size()[0], self.num_outputs).zero_()
        act_onehot.scatter_(1, act.data, 1)

        act_pred = self.inverse_model(rep_old, rep_new)
        state_pred = self.forward_model(Variable(rep_old.data), Variable(act_onehot))

        forward_loss = (Variable(rep_new.data) - state_pred).pow(2).sum(1).mean(0)
        act = act.squeeze()
        inverse_loss = F.cross_entropy(act_pred, act)

        return forward_loss, (1.0 - beta) * inverse_loss + beta * forward_loss, inverse_loss.data
