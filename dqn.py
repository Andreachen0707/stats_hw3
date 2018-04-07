##########################################
# Stat232A&CS266A Project 3:
# Solving CartPole with Deep Q-Network
# Author: Feng Gao
##########################################

import argparse
import gym
import random
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import deque
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T


parser = argparse.ArgumentParser(description='DQN_AGENT')
parser.add_argument('--epochs', type=int, default=200, metavar='E',
                    help='number of epochs to train (default: 300)')
parser.add_argument('--batch-size', type=int, default=32, metavar='B',
                    help='batch size for training (default: 32)')
parser.add_argument('--memory-size', type=int, default=10000, metavar='M',
                    help='memory length (default: 10000)')
parser.add_argument('--max-step', type=int, default=250,
                    help='max steps allowed in gym (default: 250)')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if args.cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if args.cuda else torch.ByteTensor
Tensor = FloatTensor

env = gym.make('CartPole-v0').unwrapped

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()


global_step = 0
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
###################################################################
# Image input network architecture and forward propagation. Dimension
# of output layer should match the number of actions.
###################################################################
        # Define your network structure here:
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(448, 2)

    def forward(self, x):
                x = F.relu(self.bn1(self.conv1(x)))
                x = F.relu(self.bn2(self.conv2(x)))
                x = F.relu(self.bn3(self.conv3(x)))
                return self.head(x.view(x.size(0), -1))

class DQN_state(nn.Module):
###################################################################
# State vector input network architecture and forward propagation.
# Dimension of output layer should match the number of actions.
##################################################################
        # Define your network structure here (no need to have conv
        # block for state input):
    def __init__(self):
        super(DQN_state,self).__init__()
        self.affine1 = nn.Linear(4,128)
        self.affine2 = nn.Linear(128,2)

        
        # Define your forward propagation function here:
    def forward(self, x):
        x = F.relu(self.affine1(x))
        #action_scores = self.affine2(x)
        return self.affine2(x,view(x.size(0),-1))



class DQNagent():
    def __init__(self):
        self.model = DQN()
        self.memory = deque(maxlen=args.memory_size)
        self.gamma = 0.8
        self.epsilon_start = 1
        self.epsilon_min = 0.05
        self.epsilon_decay = 200

###################################################################
# remember() function
# remember function is for the agent to get "experience". Such experience
# should be storaged in agent's memory. The memory will be used to train
# the network. The training example is the transition: (state, action,
# next_state, reward). There is no return in this function, instead,
# you need to keep pushing transition into agent's memory. For your
# convenience, agent's memory buffer is defined as deque.
###################################################################
    def remember(self, state, action, next_state, reward):
        transition = Transition._make([state,action,next_state,reward])
        
#        self.memory.append(transition._asdict())
        self.memory.append(transition)
        #
        #self.memory.append(Transition.state)
        #self.memory.append(Transition.action)
        #self.memory.append(Transition.next_state)
        #self.memory.append(Transition.reward)
        #print(self.memory[-1])
                
                

###################################################################
# act() fucntion
# This function is for the agent to act on environment while training.
# You need to integrate epsilon-greedy in it. Please note that as training
# goes on, epsilon should decay but not equal to zero. We recommend to
# use the following decay function:
# epsilon = epsilon_min+(epsilon_start-epsilon_min)*exp(-1*global_step/epsilon_decay)
# act() function should return an action according to epsilon greedy. 
# Action is index of largest Q-value with probability (1-epsilon) and 
# random number in [0,1] with probability epsilon.
###################################################################
    def act(self, state):
                global global_step
                sample_test = random.random()
                epsilon = self.epsilon_min+(self.epsilon_start-self.epsilon_min)*math.exp(-1*global_step/self.epsilon_decay)
                global_step+=1
                if sample_test>epsilon:
                        return self.model(Variable(state,volatile=True).type(FloatTensor)).data.max(1)[1].view(1,1)
                else:
                        return LongTensor([[random.randrange(2)]])


                
###################################################################
# replay() function
# This function performs an one step replay optimization. It first
# samples a batch from agent's memory. Then it feeds the batch into 
# the network. After that, you will need to implement Q-Learning. 
# The target Q-value of Q-Learning is Q(s,a) = r + gamma*max_{a'}Q(s',a'). 
# The loss function is distance between target Q-value and current
# Q-value. We recommend to use F.smooth_l1_loss to define the distance.
# There is no return of act() function.
# Please be noted that parameters in Q(s', a') should not be updated.
# You may use Variable().detach() to detach Q-values of next state 
# from the current graph.
###################################################################
    def replay(self, batch_size):
                if len(self.memory)<batch_size:
                    return
                sample_batch = random.sample(self.memory,batch_size)
                
                sample_batch = Transition(*zip(*sample_batch))
                
                non_final_mask = ByteTensor(tuple(map(lambda s :s is not None,
                                                      sample_batch.next_state)))
                non_final_next_states = Variable(torch.cat([s for s in sample_batch.next_state
                                                       if s is not None]),
                                            volatile = True)
                state_batch = Variable(torch.cat(sample_batch.state))
                action_batch = Variable(torch.cat(sample_batch.action))
                reward_batch = Variable(torch.cat(sample_batch.reward))

                state_action_value = self.model(state_batch).gather(1,action_batch)
                next_state_values = Variable(torch.zeros(batch_size).type(Tensor))
                next_state_values[non_final_mask]=self.model(non_final_next_states).max(1)[0]

                next_state_values = next_state_values.detach()

 #               next_state_values.volatile = False
                
                expected_state_action_value = next_state_values *self.gamma +reward_batch
                loss = F.smooth_l1_loss(state_action_value,expected_state_action_value)
                optimizer.zero_grad()
                loss.backward()
                for param in self.model.parameters():
                    param.grad.data.clamp_(-1,1)
                optimizer.step()
 
                
                


#################################################################
# Functions 'getCartLocation' and 'getGymScreen' are designed for 
# capturing current renderred image in gym. You can directly take 
# the return of 'getGymScreen' function, which is a resized image
# with size of 3*40*80.
#################################################################

def getCartLocation():
    world_width = env.x_threshold*2
    scale = 600/world_width
    return int(env.state[0]*scale+600/2.0)

def getGymScreen():
    screen = env.render(mode = 'rgb_array').transpose(
            (2,0,1))
    screen = screen[:, 160:320]
    view_width = 320
    cart_location = getCartLocation()
    if cart_location < view_width//2:
        slice_range = slice(view_width)
    elif cart_location > (600-view_width//2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width//2, cart_location+view_width//2)
    screen = screen[:, :, slice_range]
    screen = np.ascontiguousarray(screen, dtype=np.float32)/255
    screen = torch.FloatTensor(screen)
    return resize(screen).unsqueeze(0)


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(durations)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 30:
        means = durations_t.unfold(0, 30, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(29), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())
 



#def main():
env = gym.make('CartPole-v0').unwrapped
env._max_episode_steps = args.max_step
env.reset()
plt.figure()
plt.imshow(getGymScreen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
           interpolation='none')
plt.title('Example extracted screen')
plt.show()
    
print('env max steps:{}'.format(env._max_episode_steps))
steps_done = 0
agent = DQNagent()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, agent.model.parameters()), lr=1e-3)
durations = []
    ################################################################
    # training loop
    # You need to implement the training loop here. In each epoch, 
    # play the game until trial ends. At each step in one epoch, agent
    # need to remember the transitions in self.memory and perform
    # one step replay optimization. Use the following function to 
    # interact with the environment:
    #   env.step(action)
    # It gives you infomation about next step after taking the action.
    # The return of env.step() is (next_state, reward, done, info). You
    # do not need to use 'info'. 'done=1' means current trial ends.
    # if done equals to 1, please use -1 to substitute the value of reward.
    ################################################################
for epoch in range(1, args.epochs+1):
    steps = 0
        
    ################################################################
    # Image input. We recommend to use the difference between two
    # images of current_screen and last_screen as input image.
    ################################################################
    env.reset()
    print(epoch)
    last_screen = getGymScreen()
    current_screen = getGymScreen()
    state = current_screen - last_screen
    for t in count():
        action = agent.act(state)
        _,reward,done,_ = env.step(action[0,0])
        #print(done)
        reward = Tensor([reward])
        last_screen = current_screen
        current_screen = getGymScreen()
        if not done:
            next_state = current_screen - last_screen
            steps+=1
        else:
            next_state = None
        agent.remember(state,action,next_state,reward)
        state = next_state
        agent.replay(args.batch_size)
 
  
        if done:
            durations.append(steps)
            plot_durations()
            break
                    
        
    ################################################################
    # State vector input. You can direct take observation from gym 
    # as input of agent's DQN
    ################################################################

    #state = env.reset()

    ################################################################
plt.savefig('output.pdf')
print(durations)
#env.close()



#if __name__ == "__main__":
#    main()
