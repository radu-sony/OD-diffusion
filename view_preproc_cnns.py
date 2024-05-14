
# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from resources.utils import make_cnn_gamma
import matplotlib.pyplot as plt

import numpy as np
import time

in_chans = 1
hidden_channels = 8
gamma = 0.5
weights, biases = make_cnn_gamma(gamma, hidden_channels, in_chans=in_chans)

print(weights)
print(biases)
print()

trainable = False

preproc_params = {'name':'dynamic1',
                'value': 1,
                'rggb_max': 2**14,
                'hidden_size': 32,
                'n_layers': 4,
                'n_params': 3,
                'mean': [],
                'std': [],
                'scale_test': 1,
                'gamma': gamma,
                'trainable': trainable,
                'hidden_channels': hidden_channels,
                'input_channels': in_chans}


class gamma_cnn(nn.Module):
    def __init__(self, preproc_params = []):
        super().__init__()

        print('-> Building simple gamma max layer.')

        self.gamma = preproc_params['gamma']
        trainable = preproc_params['trainable']
        rggb_max = preproc_params['rggb_max']
        self.hidden_channels = preproc_params['hidden_channels']
        self.input_channels = preproc_params['input_channels']

        self.cnn1 = nn.Sequential(nn.Conv2d(self.input_channels, 
                                            self.input_channels*self.hidden_channels, 
                                            kernel_size=1, 
                                            stride=1, 
                                            bias=True, 
                                            groups=self.input_channels),
                                  nn.ReLU())
        self.cnn2 = nn.Sequential(nn.Conv2d(self.input_channels*self.hidden_channels,
                                            self.input_channels,  
                                            kernel_size=1, 
                                            stride=1, 
                                            bias=False, 
                                            groups=self.input_channels),
                                  nn.ReLU())
        
        print(self.cnn1[0].weight.size())
        print(self.cnn1[0].bias.size())
        print(self.cnn2[0].weight.size())

        weights1 = np.ones((self.input_channels*self.hidden_channels,1,1,1), dtype=np.float32)
        weights2, biases = self.make_cnn_gamma()
        print(weights2.shape)
        print('bias', biases.shape)
        self.cnn1[0].weight = nn.Parameter(torch.ones_like(self.cnn1[0].weight))
        self.cnn1[0].bias = nn.Parameter(torch.from_numpy(biases))
        self.cnn2[0].weight = nn.Parameter(torch.from_numpy(weights2))

        self.rggb_max = torch.tensor(preproc_params['rggb_max'], requires_grad=False).cuda()

        if not trainable:
            self.cnn1[0].requires_grad = False
            self.cnn2[0].requires_grad = False
        
    def fun(self, x, gamma):
        return x**gamma

    def make_cnn_gamma(self):
        bias = np.linspace(0, 1, self.hidden_channels+1)
        bias = bias ** 2 # since gamma is 'bendy' close to 0, have more bias values close to 0.

        weights = np.zeros(self.hidden_channels)

        for i in range(len(weights)):
            w = (self.fun(bias[i+1], self.gamma)-self.fun(bias[i], self.gamma)) / (bias[i+1]-bias[i]) - np.sum(weights)
            weights[i] = w

        #weights = np.concatenate([weights]*in_chans)
        weights = np.tile(weights, (self.input_channels, 1))
        weights = np.expand_dims(np.expand_dims(weights, axis=-1), axis=-1).astype(np.float32)
        biases = np.concatenate([bias[:-1]]*self.input_channels).astype(np.float32) * -1

        return weights, biases

    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        return x   
    

model = gamma_cnn(preproc_params)
# cuda_no = 3
# device = torch.device(f'cuda:{cuda_no}')  # GPU 1 is 'cuda:1'
# torch.cuda.set_device(device)

n = 10
rggb_max = 2**14
x0 = np.linspace(0, 1, n)
x = x0
x = np.repeat(x, in_chans).reshape(-1, in_chans)
print('x', x.shape)

print([x[i,0] for i in range(n)])

x = np.expand_dims(np.expand_dims(x, -1), -1)
x = torch.from_numpy(x).float()
# print('x', x.shape)

# print('weights n biases')
print(' ')
print(model.cnn1[0].weight.size())
print(model.cnn1[0].bias.size())
print(model.cnn2[0].weight.size())

# exit()
# print(x.shape)

y = model(x)
print(y.size())
for j in range(n):
    #print([y[i,j,0,0].item() for i in range(n)])
    plt.plot(x0, [y[j,0,0,0].item() for j in range(n)])

stamp = str(int(time.time()))
plt.savefig(f'./outputs/{stamp}.jpg')