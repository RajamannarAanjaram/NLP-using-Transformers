"""
Created on Wed Sep 22 13:31:57 2021

@author: Rajamannar.Aanjaram
"""

import torch.nn as nn

def XOR_weights_init(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 1)
