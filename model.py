import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #print("init conv")
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        #print("init lin")
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)

rep_size = 32 * 3 * 3

class ActorCritic(torch.nn.Module):

    def __init__(self, num_inputs, action_space):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.lstm = nn.LSTMCell(32 * 3 * 3, 256)

        self.num_outputs = action_space
        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, self.num_outputs)

        ############

        self.encoder = nn.Sequential(
            nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1),
            torch.nn.ELU(),
            #torch.nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            torch.nn.ELU(),
            #torch.nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            torch.nn.ELU(),
            #torch.nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            torch.nn.ELU(),
            #torch.nn.BatchNorm2d(32),
        )

        #self.bn = torch.nn.BatchNorm1d(rep_size)

        lin1 = nn.Linear(32 * 3 * 3 * 2, 256)
        lin2 = nn.Linear(256, self.num_outputs)

        self.act_pred = nn.Sequential(
            lin1,
            torch.nn.ReLU(),
            lin2,
            #torch.nn.Softmax()
        )

        lin3 = nn.Linear(32 * 3 * 3 + self.num_outputs , 256)
        lin4 = nn.Linear(256, 32 * 3 * 3)

        self.state_pred = nn.Sequential(
            lin3,
            torch.nn.ReLU(),
            lin4,
        )

        self.apply(weights_init)

        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        lin1.weight.data = normalized_columns_initializer(lin1.weight.data, 0.01)
        lin1.bias.data.fill_(0)
        lin2.weight.data = normalized_columns_initializer(lin2.weight.data, 0.01)
        lin2.bias.data.fill_(0)
        lin3.weight.data = normalized_columns_initializer(lin3.weight.data, 0.01)
        lin3.bias.data.fill_(0)
        lin4.weight.data = normalized_columns_initializer(lin4.weight.data, 0.01)
        lin4.bias.data.fill_(0)

        self.train()

    def inverse_model(self, rep_old, rep_new):
        x = torch.cat([rep_old, rep_new], 1)
        logits = self.act_pred(x)
        return logits

    def forward_model(self, rep_old, act):
        x = torch.cat([rep_old, act], 1)
        pred = self.state_pred(x)
        return pred

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = x.view(-1, rep_size)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        return self.critic_linear(x), self.actor_linear(x), (hx, cx)

    def icm(self, state_old, act, state_new):
        forward_loss_wt = 0.2

        rep_old = self.encoder(state_old).view(-1, rep_size)
        rep_new = self.encoder(state_new).view(-1, rep_size)

        act_onehot = torch.FloatTensor(act.size()[0], self.num_outputs).zero_()
        act_onehot.scatter_(1, act.data, 1)

        act_pred = self.inverse_model(rep_old, rep_new)
        state_pred = self.forward_model(rep_old, Variable(act_onehot))

        forward_loss = 0.5 * ((rep_new - state_pred).pow(2.0)).mean() * rep_size
        act = act.squeeze()
        #print(act_onehot, act_pred)
        inverse_loss = F.cross_entropy(act_pred, act, size_average=True)
        inv_acc = torch.mean((torch.max(act_pred, 1)[1] == act).float())

        return (1.0 - forward_loss_wt) * inverse_loss + forward_loss_wt * forward_loss, inverse_loss.data, forward_loss.data, inv_acc.data

    def calc_bonus(self, state_old, act, state_new):
        state_old = Variable(state_old.unsqueeze(0))
        state_new = Variable(state_new.unsqueeze(0))

        rep_old = self.encoder(state_old).view(-1, rep_size)
        rep_new = self.encoder(state_new).view(-1, rep_size)

        act_onehot = Variable(torch.FloatTensor(act.size()[0], self.num_outputs).zero_())
        act_onehot.scatter_(1, act, 1)

        prediction_beta = 0.01

        state_pred = self.forward_model(rep_old, act_onehot)
        forward_loss = 0.5 * ((rep_new - state_pred).pow(2.0)).mean() * rep_size

        return forward_loss.data * prediction_beta
