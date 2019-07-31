# Implementation of Actor-Critic algorithm:
# Based on the proposal of Richard Sutton, pag. 332
import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
#from collections import deque
import pickle

ID_EXECUTION        =   'XS003-AC'
TRAINING            =   True

NUMBER_EPISODES     =   500000
EPISODE_MAX_LEN     =   10000

GAMMA               =   0.99
LEARNING_RATE_P     =	1e-4
LEARNING_RATE_V     =   1e-3

## Define Actor critic network by MLP 
class PolicyNetwork(nn.Module):
    def __init__(self, space_n, action_n):
        super(PolicyNetwork, self).__init__()
        self.lin1   =   nn.Linear(space_n, 64)
        self.lin2   =   nn.Linear(64, action_n)
        self.softmax    =   nn.Softmax(dim=-1)

    def forward(self, obs):
        x           =   torch.tanh(self.lin1(obs))
        x           =   self.softmax(self.lin2(x))
        return x

class ValueNetwork(nn.Module):
    def __init__(self, space_n):
        super(ValueNetwork, self).__init__()
        self.lin1   =   nn.Linear(space_n, 64)
        self.lin2   =   nn.Linear(64, 1)
    
    def forward(self, obs):
        x           =   torch.tanh(self.lin1(obs))
        x           =   self.lin2(x)
        return  x

def ActorCritic(env:gym.Env):
    device      =   torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    set_actions =   env.action_space
    space_obs   =   env.observation_space.shape[0]
    policynet   =   PolicyNetwork(space_obs, set_actions.n).to(device)
    valuenet    =   ValueNetwork(space_obs).to(device)

    print('Policy Network:\n', policynet)
    print('Value Network:\n', valuenet)

    # Optimizer:
    optimizer   =   optim.Adam([{'params':policynet.parameters(), 'lr':LEARNING_RATE_P},
                                {'params':valuenet.parameters(), 'lr': LEARNING_RATE_V}])

    for episode in range(1, NUMBER_EPISODES + 1):
        ob      =   env.reset()
        cum_rw  =   0

        done    =   False

        I       =   1
        while not done:
            probs   =   policynet(torch.tensor(ob, dtype=torch.float32, device=device))
            with torch.no_grad():
                distr   =   Categorical(probs)
                action  =   distr.sample().item()
            
            obnew, rw, done, _  =   env.step(action)

            v_target    =   valuenet(torch.tensor(obnew, dtype=torch.float32, device=device)).detach()
            v_expected  =   valuenet(torch.tensor(ob, dtype=torch.float32, device=device))
            deltav      =   rw - v_expected + (GAMMA * v_target) if not done else 0

            lossv       =   deltav

            lossp       =   -I * deltav.detach().item() * torch.log(probs[action])

            loss        =   lossv + lossp

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            I           =   I * GAMMA
            ob          =   obnew



