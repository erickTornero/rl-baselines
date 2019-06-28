import gym
import random
# Deque to memory replay
from collections import deque
# Pytorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# Import CV to preprocess image
import cv2
import sys

import numpy as np

BATCH_SIZE          =   32
MEMORY_REPLAY_LEN   =   10000
NUMBER_EPISODES     =   1000
EPISODE_MAX_LEN     =   10000
EPSILON_START       =   1.0
EPSILON_END         =   0.1
GAMMA               =   0.99

LEN_DECAYING        =   1000000.0
DECAY_RATE          =   (-EPSILON_END + EPSILON_START)/LEN_DECAYING

env     =   gym.make('Pong-v0')


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

        st = torch.from_numpy(np.vstack([e[0] for e in batch])).float().to(device)
        rw = torch.from_numpy(np.vstack([e[1] for e in batch])).float().to(device)
        ac = torch.from_numpy(np.vstack([e[2] for e in batch])).float().to(device)
        ns = torch.from_numpy(np.vstack([e[3] for e in batch])).float().to(device)
        do = torch.from_numpy(np.vstack([float(e[4]) for e in batch])).float().to(device)

        return (st, rw, ac, ns, do)
    
    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    # Output in an cnn layer:
    #row, # cols = (ncols_in - kernel_sz + 2*padd)/stride + 1
    def __init__(self, n_act):
        super(DQN, self).__init__()
        self.conv1      =   nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        # out: [(84 - 8 + 2*0)/4 + 1]-->(20, 20) x 32
        self.conv2      =   nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        # out: [(20 - 4 + 2*0)/2 + 1]-->(9, 9) x 64
        self.conv3      =   nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        # out: [(9 - 3 + 2*0)/1 + 1]--> (7, 7) x 64
        # Flat: 7 * 7 * 64 = 3136
        self.lin1       =   nn.Linear(7*7*64, 512)
        self.lin_out    =   nn.Linear(512, n_act)

    def forward(self, obs):
        x       =       F.relu(self.conv1(obs))
        x       =       F.relu(self.conv2(x))
        x       =       F.relu(self.conv3(x))
        ### Flattern the output
        x       =       x.view(-1, 7 * 7 * 64)
        x       =       F.relu(self.lin1(x))
        x       =       self.lin_out(x)
        return x

#def ToGrayScale(img):
#    sh  =   img.shape
#    if sh.shape != 3:
#        print('Dimension Expected 3!')
#        return sh
#    if sh.shape[2]!=3:
#        print('Expected RGB channels!')
#        return sh
#    
#    gray = img[:,:,0]*0.2989 + img[:,:,1]*0.5870 + img[:,:,2]*0.1140
#
#    return gray

def preprocessing(img):
    img     =   img[33:195]  ## Cropping just PONG game
    img     =   cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img     =   cv2.resize(img, dsize=(84, 84), interpolation= cv2.INTER_LINEAR)
    return img.astype(np.float32)/255.0

class Observations:
    def __init__(self):
        self.obs = deque([np.zeros((84, 84), dtype=np.float32)]*4, maxlen=4)

    def Push(self, newobs):
        self.obs.append(newobs)
    
    def Get(self):
        return np.asarray(self.obs, dtype=np.float32)
    def PushAndGet(self, newobs):
        self.Push(newobs)
        return self.Get()

def train_dqn(env):
    ob      =   env.reset()
    e_greed =   EPSILON_START
    set_actions =   env.action_space

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('DEVICE> ', device)
    GeneralCounter  =   0.0
    FramesCounter   =   0.0
    dqn = DQN(set_actions.n)
    #dqn.float()
    # Pass dqn to device
    dqn.to(device)
    #dqn.p
    print('Summary>\n', dqn)

    memory_replay  =   MemoryReplay(MEMORY_REPLAY_LEN)
    optimizer       =   optim.Adam(dqn.parameters(), lr=0.0005)

    for episode in range(NUMBER_EPISODES):
        statehandler    =   Observations()
        ob              =   env.reset()
        ob              =   preprocessing(ob)
        xin             =   statehandler.PushAndGet(ob)  
        cum_rw          =   0

        print('Episode> ', episode+1, 'e_greed> ', e_greed)    
        for t in range(EPISODE_MAX_LEN):        
            #env.render()
            if GeneralCounter % 4 == 0:
                if random.random() < e_greed:
                    action  =   set_actions.sample()
                else:
                    #xin = torch.from_numpy(xin).to(device)
                    qvalues =   dqn(torch.tensor(xin, dtype=torch.float32, device=device).unsqueeze(0))
                    action  =   qvalues.argmax().item()

                ob, rw, done, _     =   env.step(action)
                ob                  =   preprocessing(ob)
                xnew                =   statehandler.PushAndGet(ob)
                
                
                
                memory_replay.push((xin, rw, action, xnew, done))

                xin = xnew
                ##
##
                batch               =   memory_replay.sample(BATCH_SIZE, device)
                
                #print(batch[2].unsqueeze(1).long().shape)
                #print(dqn(batch[0]).shape)
                Qpred   =   batch[1] + GAMMA * dqn(batch[3].unsqueeze(0)).max(dim=1)[0] * (1.0 - batch[4])
                Qtarg   =   torch.gather(dqn(batch[0].unsqueeze(0)), 1, batch[2].unsqueeze(1).long()).squeeze(1)
                
                #print(batch[2])
                
                loss = F.mse_loss(Qpred, Qtarg)
                
                print('len>', len(memory_replay), 'bytes> ', sys.getsizeof(memory_replay))
                #
                #for dt in batch:
                #    #print(len(batch))
                #    qvalues_    =  dqn(torch.from_numpy(dt[0]).unsqueeze(0).float())
                #    if dt[4]:
                #        losslist.append(torch.tensor(dt[2]) - qvalues_.data[0, dt[1]])
                #    else:
                #        qvalues = dqn(torch.from_numpy(dt[3]).unsqueeze(0).float())
                #        losslist.append(torch.tensor(rw) + GAMMA * qvalues.max() - qvalues_.data[0, dt[1]])   
                #
                #loss = 0
                #for ls in losslist:
                #    loss = loss + ls*ls
                #loss = torch.FloatTensor(loss)
                #loss = (loss**2)
                ##if len(loss) < 2:
                #loss = torch(loss)
                ##else:
                ##loss = loss.mean()
                #loss = loss.mean()
                #print(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                #print(y_pred)

                if FramesCounter < LEN_DECAYING:
                    e_greed = e_greed - DECAY_RATE

                FramesCounter   = FramesCounter + 1
                
                #print(e_greed)
            else:
                ob, rw, done, _ = env.step(action)

            cum_rw = cum_rw + rw
            if done:
                break

            GeneralCounter = GeneralCounter + 1.0
        if GeneralCounter % 5 == 0:
            print('Cum Reward> ', cum_rw)
            print('Frames counted>', FramesCounter)
            print('len>', len(memory_replay))
            
        cum_rw = 0
        if GeneralCounter % 20 == 0:
            torch.save( dqn.state_dict(), 'dqn_saved_model.pth' )

    torch.save( dqn.state_dict(), 'dqn_saved_model.pth' )

train_dqn(env)




#for _ in range(1000):
#    env.render()
#    action  =   set_actions.sample()
#    obs, rw, done, info     =   env.step(action)
#
#    if done:
#        obs =   env.reset()

env.close()
