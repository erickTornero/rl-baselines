import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import pickle

ID_EXECUTION        =   'XS001-DUEL'
TRAINING            =   True
USE_BATCH_NORM      =   False

TARGET_UPDATE       =   2
BATCH_SIZE          =   32
MEMORY_REPLAY_LEN   =   200000
REPLAY_START_SIZE   =   100

NUMBER_EPISODES     =   500000
EPISODE_MAX_LEN     =   10000
EPSILON_START       =   1.0
EPSILON_END         =   0.1
GAMMA               =   0.99
LEARNING_RATE       =	1e-4
TAU                 =   0.001
LEN_DECAYING        =   1000.0
DECAY_RATE          =   (-EPSILON_END + EPSILON_START)/LEN_DECAYING

class MemoryReplay:
    def __init__(self, m_len):
        self.memory     =   deque([], maxlen=m_len)

    def push(self, tuples):
        self.memory.append(tuples)

    def sample(self, batch_sz, device):
        if len(self.memory) < batch_sz:
            batch   =   random.sample(self.memory, len(self.memory))
        else:
            batch   =   random.sample(self.memory, batch_sz)

        ## st = torch.tensor( np.vstack([e[0] for e in batch]) ).to( device )
        ## rw = torch.tensor( np.vstack([e[1] for e in batch]) ).to( device )
        ## ac = torch.tensor( np.vstack([e[2] for e in batch]) ).to( device )
        ## ns = torch.tensor( np.vstack([e[3] for e in batch]) ).to( device )
        ## do = torch.tensor( np.vstack([float(e[4]) for e in batch]) ).to( device )

        st = torch.from_numpy(np.vstack([[e[0]] for e in batch])).float().to(device)
        rw = torch.from_numpy(np.vstack([e[1] for e in batch])).float().to(device)
        ac = torch.from_numpy(np.vstack([e[2] for e in batch])).long().to(device)
        ns = torch.from_numpy(np.vstack([[e[3]] for e in batch])).float().to(device)
        do = torch.from_numpy(np.vstack([float(e[4]) for e in batch])).float().to(device)

        return (st, rw, ac, ns, do)
    def __len__(self):
        return len(self.memory)


class DuelDQSimpleNet(nn.Module):
    def __init__(self, space_n, action_n):
        super(DuelDQSimpleNet, self).__init__()
        self.action_n   =   action_n
        self.lin1  =   nn.Linear(space_n, 64)
        #self.lin1a  =   nn.Linear(space_n, 64)
        self.lin2a  =   nn.Linear(64, action_n)
        
        #self.lin1q  =   nn.Linear(space_n, 64)
        self.lin2q  =   nn.Linear(64, 1)

    def forward(self, obs):
        x   =   torch.tanh(self.lin1(obs))
        xa   =   self.lin2a(x)
        
        #xv  =   torch.tanh(self.lin1q(obs))
        xv  =   self.lin2q(x)

        xv  =   xv  -   torch.mean(xa, dim=1, keepdim=True)
        xv  =   xv.repeat(1, self.action_n)

        #ad_r    =   xv - torch.mean(xv, dim=1, keepdim=True)

        

        return xa  + xv

def softupdatenetwork(targetnet, fromnet, tau):
    for tar_par, par in zip(targetnet.parameters(), fromnet.parameters()):
        tar_par.data.copy_(par.data*tau + tar_par.data*(1.0-tau))

# Copy parameters of a network
def copynetwork(targetnet, fromnet):
    for tar_par, par in zip(targetnet.parameters(), fromnet.parameters()):
        tar_par.data.copy_(par.data)


def train_duelqlearner(env):
    ob  =   env.reset()
    e_greed =   EPSILON_START
    set_actions =   env.action_space

    plotting_rw =   list()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('DEVICE> ', device)
    GeneralCounter  =   0
    FramesCounter   =   0

    duelqlearner        =   DuelDQSimpleNet(env.observation_space.shape[0], set_actions.n)
    target_duelqlearner =   DuelDQSimpleNet(env.observation_space.shape[0], set_actions.n)

    copynetwork(target_duelqlearner, duelqlearner)
    duelqlearner.to(device)
    target_duelqlearner.to(device)

    loss_print      =   0
    memory_replay   =   MemoryReplay(MEMORY_REPLAY_LEN)
    optimizer       =   optim.Adam(duelqlearner.parameters(), lr=LEARNING_RATE)
    
    list_reward     =   deque([], maxlen=100)
    action = 0

    for episode in range(NUMBER_EPISODES):
        ob          =   env.reset()
        cum_rw      =   0

        for t in range(EPISODE_MAX_LEN):
            # Every time take an action
            if random.random() < e_greed:
                action  =   set_actions.sample()
            else:
                with torch.no_grad():
                    duelqlearner.eval()
                    qvalues  =   duelqlearner(torch.tensor(ob, dtype=torch.float32, device=device).unsqueeze(0))
                    action      =   qvalues.argmax().item()
                    duelqlearner.train()
            
            obnew, rw, done, _  =   env.step(action)
            memory_replay.push((ob, rw, action, obnew, done))

            ob  =   obnew

            if FramesCounter > REPLAY_START_SIZE:
                batch   =   memory_replay.sample(BATCH_SIZE, device)

                # Computing of Q Target
                with torch.no_grad():
                    Qtarg   =   batch[1].squeeze(1) + GAMMA * target_duelqlearner(batch[3]).max(dim=1)[0].detach() * (1.0 - batch[4].squeeze(1))

                # Computing of Q Values
                # First compute the ponderate qvalues:

                Qpred   =   torch.gather(duelqlearner(batch[0]), 1, batch[2]).squeeze(1)

                loss_fn =   nn.MSELoss()
                loss    =   loss_fn(Qpred, Qtarg)
                loss_print  =   loss.item()

                e_greed =   max(EPSILON_END, e_greed - DECAY_RATE)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if FramesCounter % TARGET_UPDATE == 0:
                    #print('Updating Target Network ...')
                    softupdatenetwork(target_duelqlearner, duelqlearner, TAU)
                if ((episode+1)%10==0) & (t%10==0):
                    print('[{}, {}] \t\t {}\t\t{}'.format(episode+1, t, loss_print, e_greed))
                
            FramesCounter = FramesCounter + 1
            cum_rw  =   cum_rw + rw
            GeneralCounter = GeneralCounter + 1

            if done:
                break
        list_reward.append(cum_rw)
        if (episode+1)%100 == 0:
            meanrw = sum(list_reward)/len(list_reward)
            print('**************************\nMean {} Last Score>\t{}'.format(len(list_reward), meanrw))
            plotting_rw.append(meanrw)

        #    print('Frames counted>\t\t', FramesCounter)
        #    print('len>\t\t\t', len(memory_replay))
        #    print('loss>\t\t\t', loss_print)
        #    print('===============================================')
        
        if episode % 1000 == 0:
            print('Saving Model...')
            torch.save( duelqlearner.state_dict(), ID_EXECUTION+'-model.pth' )
            fichero = open(ID_EXECUTION+'-rw.pckl', 'wb')
            pickle.dump(plotting_rw, fichero)
            fichero.close()
            del(fichero)

        cum_rw = 0
    torch.save( duelqlearner.state_dict(), ID_EXECUTION+'-model.pth' )

env =   gym.make('CartPole-v0')

if TRAINING == True:
    train_duelqlearner(env)

env.close()
