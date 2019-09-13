# Utils for breakout game enviroment

import numpy as np
import matplotlib.pyplot as plt
from IPython import display

import gym
import torch

class Breakout:
    def __init__(self, game='BreakoutDeterministic-v4', cuda=False):
        self.env = gym.make(game)
        self.cuda = cuda
        
        self.Reset()
        
    def Reset(self):
        self.env.reset()
        self.previous_state = self.ImageToTensor(self.env.render(mode='rgb_array'))
        self.current_state = self.previous_state
    
    #returns delta between current and previous state for network and forgots previous_state
    def GetNextInput(self):
        delta = self.current_state - self.previous_state
        self.previous_state = self.current_state
        return torch.autograd.Variable(delta, requires_grad=False)
    
    # returns reward
    def MakeStep(self, action):
        current_state, reward, done, _ = self.env.step(action)
        self.current_state = self.ImageToTensor(current_state)
        return reward, done
        
    def GetSummaryScore(self, gamer, seed=None, max_screens=int(1e5), visualizate=False):
        self.Reset()
        if not seed is None:
            self.env.seed(seed)
            
        summ_reward = 0
        self.MakeStep(1)
        for num in range(max_screens):
            _, next_action = torch.max(gamer(self.GetNextInput()).data, 1)
            next_action = next_action[0]
            reward, done = self.MakeStep(next_action)
            summ_reward += reward
            if visualizate:
                self.ShowState(title=str(num))
            if done:
                break
        return summ_reward
        
    def ImageToTensor(self, arr):
        result = None
        if self.cuda:
            result = torch.Tensor(arr).cuda()
        else:
            result = torch.Tensor(arr)
        result = torch.stack([result])
        result = torch.transpose(result, 1, 3)
        result = torch.transpose(result, 2, 3)
        return result
            
    def ShowState(self, title=""):
        plt.figure(3)
        plt.clf()
        plt.imshow(self.env.render(mode='rgb_array'))
        plt.title(title)
        plt.axis('off')

        display.clear_output(wait=True)
        display.display(plt.gcf())