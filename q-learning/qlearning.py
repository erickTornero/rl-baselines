import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import pickle

ID_EXECUTION        =   'XS008'
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

class MlpQLearner(nn.Module):
    def __init__(self, space_n, action_n):
        super(MlpQLearner, self).__init__()
        self.lin1   =   nn.Linear(space_n, 20)
        self.lin2   =   nn.Linear(20,80)
        self.lin3   =   nn.Linear(80, 20)
        self.linout =   nn.Linear(20, action_n)
    
    def forward(self, obs):
        x   =   F.relu(self.lin1(obs))
        x   =   F.relu(self.lin2(x))
        x   =   F.relu(self.lin3(x))
        x   =   self.linout(x)
        return  x
    
class SimpleMLP(nn.Module):
    def __init__(self, space_n, action_n):
        super(SimpleMLP, self).__init__()
        self.lin1   =   nn.Linear(space_n, 64)
        self.lin2   =   nn.Linear(64, action_n)
    def forward(self, obs):
        x           =   torch.tanh(self.lin1(obs))
        x           =   self.lin2(x)
        return x

class MlpQLearnerBN(nn.Module):
    def __init__(self, space_n, action_n):
        super(MlpQLearnerBN, self).__init__()
        self.lin1   =   nn.Linear(space_n, 20)
        self.bn1    =   nn.BatchNorm1d(20)
        self.lin2   =   nn.Linear(20,80)
        self.bn2    =   nn.BatchNorm1d(80)
        self.lin3   =   nn.Linear(80, 20)
        self.bn3    =   nn.BatchNorm1d(20)
        self.linout =   nn.Linear(20, action_n)
    
    def forward(self, obs):
        x   =   F.relu(self.bn1(self.lin1(obs)))
        x   =   F.relu(self.bn2(self.lin2(x)))
        x   =   F.relu(self.bn3(self.lin3(x)))
        x   =   self.linout(x)
        return  x


def softupdatenetwork(targetnet, fromnet, tau):
    for tar_par, par in zip(targetnet.parameters(), fromnet.parameters()):
        tar_par.data.copy_(par.data*tau + tar_par.data*(1.0-tau))

# Copy parameters of a network
def copynetwork(targetnet, fromnet):
    for tar_par, par in zip(targetnet.parameters(), fromnet.parameters()):
        tar_par.data.copy_(par.data)

def train_qlearner(env):
    ob          =   env.reset()
    e_greed     =   EPSILON_START
    set_actions =   env.action_space

    plotting_rw =   list()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('DEVICE> ', device)
    GeneralCounter  =   0
    FramesCounter   =   0

    if USE_BATCH_NORM==True:
        qlearner        =   MlpQLearnerBN(env.observation_space.shape[0], set_actions.n)
        target_qlearner =   MlpQLearnerBN(env.observation_space.shape[0], set_actions.n)
        qlearner = qlearner.train()
    else:
        qlearner        =   SimpleMLP(env.observation_space.shape[0], set_actions.n)
        target_qlearner =   SimpleMLP(env.observation_space.shape[0], set_actions.n)

    
    copynetwork(target_qlearner, qlearner)
    qlearner.to(device)
    target_qlearner.to(device)

    loss_print      =   0
    memory_replay   =   MemoryReplay(MEMORY_REPLAY_LEN)
    optimizer       =   optim.Adam(qlearner.parameters(), lr=LEARNING_RATE) # CAMBIAR POR Adam
    #optimizer       =   optim.RMSprop(qlearner.parameters(), lr=LEARNING_RATE, momentum=0.95)
    list_reward     =   deque([], maxlen=100)
    action = 0
    for episode in range(NUMBER_EPISODES):

        ob              =   env.reset() 
        cum_rw          =   0
        if (episode+1)%1000==0:
            print('Episode>\t', episode+1, 'e_greed> ', e_greed)
        for t in range(EPISODE_MAX_LEN): 
            ## if GeneralCounter % 1 == 0:
            if True:
                if random.random() < e_greed:
                    action  =   set_actions.sample()
                else:
                    #xin = torch.from_numpy(xin).to(device)
                    with torch.no_grad():
                        qlearner.eval()
                        qvalues =   qlearner(torch.tensor(ob, dtype=torch.float32, device=device).unsqueeze(0))
                        action  =   qvalues.argmax().item()
                        qlearner.train()

                obnew, rw, done, _     =   env.step(action)

                memory_replay.push((ob, rw, action, obnew, done))
                ob                      =   obnew
                if FramesCounter > REPLAY_START_SIZE:
                    batch               =   memory_replay.sample(BATCH_SIZE, device)
                    with torch.no_grad() :
                        Qtarg   =   batch[1].squeeze(1) + GAMMA * target_qlearner(batch[3]).max(dim=1)[0].detach() * (1.0 - batch[4].squeeze(1))

                    Qpred   =   torch.gather(qlearner(batch[0]), 1, batch[2].long()).squeeze(1)

                    # Training steps
                    loss_fn     =   nn.MSELoss()  # CAMBIAR A mse loss
                    #loss        =   (Qtarg - Qpred)
                    #loss        =   loss.clamp(-20.0,20.0)
                    ##print(loss.shape)
                    #loss        =   loss**2
                    #loss        =   loss.mean()
                    loss        =   loss_fn(Qpred, Qtarg)
                    loss_print  =   loss.item()
                    #print('loss> ', loss_print)
                    ## if e_greed > EPSILON_END:
                    ##     e_greed = e_greed - DECAY_RATE

                    e_greed = max( EPSILON_END, e_greed - DECAY_RATE ) # CAMBIAR A EXPONENTIAL DECAY 
                    
                    # Optimizing network
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if FramesCounter % TARGET_UPDATE == 0:
                        #print('Updating Target Network ...')
                        softupdatenetwork(target_qlearner, qlearner, TAU)
                    if ((episode+1)%10==0) & (t%10==0):
                        print('[{}, {}] \t\t {}\t\t{}'.format(episode+1, t, loss_print, e_greed))
                
                FramesCounter = FramesCounter + 1
            
            else:
                 obnew, rw, done, _ = env.step(action)
            
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
            torch.save( qlearner.state_dict(), ID_EXECUTION+'-model.pth' )
            fichero = open(ID_EXECUTION+'-rw.pckl', 'wb')
            pickle.dump(plotting_rw, fichero)
            fichero.close()
            del(fichero)

        cum_rw = 0
    
    torch.save( qlearner.state_dict(), ID_EXECUTION+'-model.pth' )


def testqlearner(env, name, n_episodes):
    set_actions =   env.action_space
    qlearner    =   SimpleMLP(env.observation_space.shape[0], set_actions.n)
    qlearner.load_state_dict(torch.load(name))
    qlearner.eval()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('DEVICE> ', device)
    print(qlearner)
    qlearner.to(device)
    import matplotlib.pyplot as plt
    from IPython import display
    for ep in range(1, n_episodes+1):
        ob              =   env.reset() 
        cum_rw          =   0

        for t in range(EPISODE_MAX_LEN): 
            #action = set_actions.sample()
            env.render()
            with torch.no_grad():
                qvalues =   qlearner(torch.tensor(ob, dtype=torch.float32, device=device).unsqueeze(0))
                action  =   qvalues.argmax().item()
                #print(action)
            ob, rw, done, _     =   env.step(action)
            cum_rw = cum_rw + rw
            if done:
                break
            
        print(cum_rw)
        cum_rw = 0
        

            


env =   gym.make('CartPole-v0')
if TRAINING == True:
    train_qlearner(env)
else:
    testqlearner(env, ID_EXECUTION + '-model.pth',50)
env.close()

            

                    

