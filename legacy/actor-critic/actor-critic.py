# Implementation of Actor-Critic algorithm:
# Based on the proposal of Richard Sutton, pag. 332
import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
import pickle

ID_EXECUTION        =   'XS002-AC'
TRAINING            =   True

NUMBER_EPISODES     =   500000
EPISODE_MAX_LEN     =   10000
MAX_PATH_LEN        =   1000

GAMMA               =   0.99
LEARNING_RATE_P     =	1e-3
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

    print('Device: >', device)
    print('Policy Network:\n', policynet)
    print('Value Network:\n', valuenet)

    # Optimizer:
    optimizer   =   optim.Adam([{'params':policynet.parameters(), 'lr':LEARNING_RATE_P},
                                {'params':valuenet.parameters(), 'lr': LEARNING_RATE_V}])

    print('Optimizer:\n', optimizer)
    plotting_rw     =   list()
    for episode in range(1, NUMBER_EPISODES + 1):
        ob, _      =   env.reset()
        cum_rw  =   0

        done    =   False

        I       =   1

        list_reward     =   deque([], maxlen=100)
        steps = 0
        while not done and steps < MAX_PATH_LEN:
            probs   =   policynet(torch.tensor(ob, dtype=torch.float32, device=device))
            with torch.no_grad():
                distr   =   Categorical(probs)
                action  =   distr.sample().item()
            
            obnew, rw, done, _, _  =   env.step(np.int64(action))

            v_target    =   valuenet(torch.tensor(obnew, dtype=torch.float32, device=device)).detach()
            v_expected  =   valuenet(torch.tensor(ob, dtype=torch.float32, device=device))
            deltav      =   rw - v_expected + GAMMA * v_target * (1 - done)

            #lossv       =   deltav.detach() * v_expected
            lossv       =   deltav * deltav

            lossp       =   -I * deltav.detach() * torch.log(probs[action])

            loss        =   lossv + lossp

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            I           =   I * GAMMA
            ob          =   obnew
            cum_rw      =   cum_rw + rw
            steps       +=  1

        #print('{} Episode\t-->\ttotal reward>\t{}'.format(episode, cum_rw))

        list_reward.append(cum_rw)
        # Got the probability of acion chosen
        if episode % 100 == 0:
            meanrw = sum(list_reward)/len(list_reward)
            print('[{} Episode]\t-->\treward> {}\t'.format(episode, meanrw))
            plotting_rw.append(meanrw)
            #data_epsiode.append(Transition(ob, action, rw, obnew, done))
        cum_rw = 0

        if episode %500 == 0:
            # Saving network
            print('**Saving weights of Network and Rewards**')
            torch.save(policynet.state_dict(), ID_EXECUTION + 'POLICYNET-model.pth')
            torch.save(valuenet.state_dict(), ID_EXECUTION + 'VALUENET-model.pth')
            # Saving List reward
            file    =   open(ID_EXECUTION+'-rw.pckl', 'wb')
            pickle.dump(plotting_rw, file)
            file.close()
            del(file)

    # End of training, save parameters
    torch.save(policynet.state_dict(), ID_EXECUTION + 'POLICYNET-model.pth')
    torch.save(valuenet.state_dict(), ID_EXECUTION + 'VALUENET-model.pth')
    file    =   open(ID_EXECUTION+'-rw.pckl', 'wb')
    pickle.dump(plotting_rw, file)
    file.close()
    del(file)


def play(env:gym.Env, n_episodes:int):
    set_actions =   env.action_space
    obs_space   =   env.observation_space.shape[0]

    device      =   torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    policynet   =   PolicyNetwork(obs_space, set_actions.n).to(device)
    #valuenet    =   ValueNetwork(obs_space).to(device)

    policynet.load_state_dict(torch.load(ID_EXECUTION+'POLICYNET-model.pth'))
    #valuenet.load_state_dict(torch.load(ID_EXECUTION+'VALUENET-model.pth'))

    for ep in range(1, n_episodes + 1):
        ob, _      =   env.reset()
        cum_rw  =   0
        done    =   False
        while not done:
            env.render()
            with torch.no_grad():
                probs   =   policynet(torch.tensor(ob, dtype=torch.float32, device=device))
                dist    =   Categorical(probs)
                action  =   dist.sample().item()

            ob, rw, done, _, _ =   env.step(action)
            cum_rw      =   cum_rw + rw
        
        print('{} Episode\t-->\tTotal reward\t{}'.format(ep, cum_rw))


env =   gym.make('CartPole-v1')
if TRAINING==True:
    ActorCritic(env)

else:
    play(env, 10)

env.close()

