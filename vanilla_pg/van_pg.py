import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque, namedtuple
import pickle

ID_EXECUTION        =   'XS009'
TRAINING            =   True

NUMBER_EPISODES     =   500000
EPISODE_MAX_LEN     =   10000

GAMMA               =   0.99
LEARNING_RATE       =	1e-4

class SimpleMLP(nn.Module):
    def __init__(self, space_n, action_n):
        super(SimpleMLP, self).__init__()
        self.lin1   =   nn.Linear(space_n, 64)
        self.lin2   =   nn.Linear(64, action_n)
        self.softmax    =   nn.Softmax(dim=-1)
    def forward(self, obs):
        x           =   torch.tanh(self.lin1(obs))
        x           =   self.softmax(self.lin2(x))
        return x



def reinforce(env):
    ob          =   env.reset()
    set_actions =   env.action_space
    loss_print  =   0
    # Torch definitions
    device      =   torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    pg_net      =   SimpleMLP(env.observation_space.shape[0], set_actions.n)
    optimizer   =   optim.SGD(pg_net.parameters(), LEARNING_RATE)
    pg_net.to(device)
    print(pg_net)
    #Transition      =   namedtuple('Transition',['S','action','reward','Snext','done'])
    for episode in range(NUMBER_EPISODES):
        ob              =   env.reset() 
        cum_rw          =   0
        data_epsiode    =   list()
        
        # Expected to hace T States, T Rewards, T actions
        states_list     =   list()
        action_list     =   list()
        reward_list     =   list()

        done            =   False
        # Generate an episode:
        while not done:
            # Perform in every action
            # choose an action:
            with torch.no_grad():
                probs_policy    =   pg_net(torch.tensor(ob, dtype=torch.float32, device=device).unsqueeze(0))
                distribution_S  =   Categorical(probs_policy)
                action          =   distribution_S.sample().item()

            obnew, rw, done, _  =   env.step(action)
            # T-1
            states_list.append(ob)
            action_list.append(action)
            # T + 1
            reward_list.append(rw)
            ob                  =   obnew
        
        print(len(states_list))
        print(len(action_list))
        print(len(reward_list))
        break
            #data_epsiode.append(Transition(ob, action, rw, obnew, done))


env =   gym.make('CartPole-v0')

reinforce(env)
