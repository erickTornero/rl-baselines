# Implementation of reinforce algorithm:
# Based on the proposal of Richard Sutton
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

ID_EXECUTION        =   'XS003-VPG'
TRAINING            =   True

NUMBER_EPISODES     =   500000
EPISODE_MAX_LEN     =   10000

GAMMA               =   0.99
LEARNING_RATE       =	1e-3

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
    optimizer   =   optim.Adam(pg_net.parameters(), LEARNING_RATE)
    pg_net.to(device)
    print(pg_net)
    list_reward     =   deque([], maxlen=100)
    plotting_rw     =   list()
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
            cum_rw              =   cum_rw + rw
        
        rewards =   np.asarray(reward_list, dtype=np.float32)
        # Training
        # Compute G from step 0 to T - 1
        
        n_steps =   len(reward_list)
        G = np.zeros(n_steps, dtype=np.float32)
        
        for i in range(0, n_steps):
            factor  =   GAMMA ** (np.arange(0, n_steps - i))
            G[i]    =   (rewards[i:] * factor).sum()
        
        #print(G)
        ## Pass to device necessary tensors
        rewards_torch   =   torch.from_numpy(G).to(device)
        rewards_torch   =   (rewards_torch - rewards_torch.mean())/rewards_torch.std()
        states_torch    =   torch.from_numpy(np.vstack([st] for st in states_list)).float().to(device)
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
        if (episode + 1) % 100 == 0:
            meanrw = sum(list_reward)/len(list_reward)
            print('[{} Episode]\t-->\treward> {}\tloss> {}\t'.format(episode + 1, meanrw, loss.item()))
            plotting_rw.append(meanrw)
            #data_epsiode.append(Transition(ob, action, rw, obnew, done))
        cum_rw = 0

        if (episode + 1)%500 == 0:
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


env =   gym.make('CartPole-v0')

reinforce(env)
