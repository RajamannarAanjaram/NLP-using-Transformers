"""
Created on Wed Sep 22 13:31:57 2021

@author: Rajamannar.Aanjaram
"""

import torch.nn as nn
import torch

class XOR(nn.Module):
    def __init__(self, input_dim=2, output_dim=1):
        super(XOR, self).__init__()
        self.lin1 = nn.Linear(input_dim, 5)
        self.lin2 = nn.Linear(5, 4)
        self.lin3 = nn.Linear(4, output_dim)
        

    def forward(self, x):
        x = torch.tanh(self.lin1(x))
        x = torch.tanh(self.lin2(x))
        x = self.lin3(x)
        return x
    