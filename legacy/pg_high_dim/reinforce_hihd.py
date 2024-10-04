# Author Erick tornero
# Based on Karphaty implementation
# Pytorch

import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, uniform
from collections import deque, namedtuple
import pickle


ID_EXECUTION        =   'XS003-PGHD'
TRAINING            =   True

NUMBER_EPISODES     =   500000
EPISODE_MAX_LEN     =   10000

GAMMA               =   0.99
LEARNING_RATE       =	1e-4

H                   =   200
D                   =   80 * 80


def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float32).ravel()


class SimpleMLP(nn.Module):
    def __init__(self, action_n):
        super(SimpleMLP, self).__init__()
        self.lin1   =   nn.Linear(D, H)
        self.lin2   =   nn.Linear(H, action_n)
        self.softmax    =   nn.Softmax(dim=-1)
    def forward(self, obs):
        x           =   torch.tanh(self.lin1(obs))
        x           =   self.softmax(self.lin2(x))
        #x           =   1.0/(1.0 + torch.exp(-x))
        return x


def reinforce(env:gym.Env):

    loss_print  =   0
    # Torch definitions
    device      =   torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    pg_net      =   SimpleMLP(action_n=2)
    optimizer   =   optim.Adam(pg_net.parameters(), LEARNING_RATE)

    pg_net.to(device)
    print(pg_net)
    list_reward     =   deque([], maxlen=100)
    plotting_rw     =   list()
    ## Sampler ..
    samplerU        =   uniform.Uniform(0.0, 1.0)

    for episode in range(NUMBER_EPISODES):
        ob, _              =   env.reset() 
        cum_rw          =   0
        data_epsiode    =   list()
        
        # Expected to hace T States, T Rewards, T actions
        states_list     =   list()
        action_list     =   list()
        reward_list     =   list()

        done            =   False
        
        prev_st         =   None
        cur_st          =   prepro(ob)
        # Generate an episode:
        while not done:
            # Perform in every action
            # choose an action:
            x_st        =   cur_st - prev_st if prev_st is not None else np.zeros(D, dtype=np.float32)
            prev_st     =   cur_st
            with torch.no_grad():
                probs_policy    =   pg_net(torch.tensor(x_st, dtype=torch.float32, device=device).unsqueeze(0))
                #action          =   2 if samplerU.sample().item < probs_policy.item() else 3
                distribution_S  =   Categorical(probs_policy)
                action          =   distribution_S.sample().item()

            obnew, rw, done, _, _  =   env.step(action+2)
            # T-1

            states_list.append(x_st)
            action_list.append(action)
            # T + 1
            reward_list.append(rw)
            ob                  =   obnew
            cum_rw              =   cum_rw + rw

        ## Rewards:
        rewards =   np.asarray(reward_list, dtype=np.float32)
        # Training
        # Compute G from step 0 to T - 1
        
        n_steps =   len(reward_list)
        G = np.zeros(n_steps, dtype=np.float32)
        
        for i in range(0, n_steps):
            factor  =   GAMMA ** (np.arange(0, n_steps - i))
            G[i]    =   (rewards[i:] * factor).sum()
        
        ## Pass to device necessary tensors
        rewards_torch   =   torch.from_numpy(G).to(device)
        rewards_torch   =   (rewards_torch - rewards_torch.mean())/rewards_torch.std()
        states_torch    =   torch.from_numpy(np.vstack([[st] for st in states_list])).to(device, dtype=torch.float32)
        gamma_torch     =   GAMMA ** torch.arange(n_steps, dtype=torch.float32, device=device)
        actions_torch   =   torch.from_numpy(np.asarray(action_list)).long().to(device)

        act_prob        =   torch.gather(pg_net(states_torch), 1, actions_torch.unsqueeze(1)).squeeze(1)

        loss            =   gamma_torch * rewards_torch * torch.log(act_prob)
        loss            =   -loss.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        list_reward.append(cum_rw)
        # Got the probability of acion chosen
        if (episode + 1) % 10 == 0:
            meanrw = sum(list_reward)/len(list_reward)
            print('[{} Episode]\t-->\treward> {}\tloss> {}\t'.format(episode + 1, meanrw, loss.item()))
            plotting_rw.append(meanrw)
            #data_epsiode.append(Transition(ob, action, rw, obnew, done))
        cum_rw = 0

        if (episode + 1)%50 == 0:
            # Saving network
            print('**Saving weights of Network and Rewards**')
            torch.save(pg_net.state_dict(), ID_EXECUTION + '-model.pth')
            # Saving List reward
            file    =   open(ID_EXECUTION+'-rw.pckl', 'wb')
            pickle.dump(plotting_rw, file)
            file.close()
            del(file)

    # End of training, save parameters
    torch.save(pg_net.state_dict(), ID_EXECUTION + '-model.pth')
    file    =   open(ID_EXECUTION+'-rw.pckl', 'wb')
    pickle.dump(plotting_rw, file)
    file.close()
    del(file)


def play(env:gym.Env):
    pass



env =   gym.make('Pong-v0')
if TRAINING==True:
    reinforce(env)
else:
    play(env, ID_EXECUTION+'-model.pth', 20)


env.close()