# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 13:31:57 2021

@author: Rajamannar.Aanjaram
"""

from src.models import XOR
from src.dataloader import XOR_weights_init
import torch
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
import torch.nn as nn
import warnings

warnings.filterwarnings('ignore')

model = XOR()

X = torch.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = torch.Tensor([0, 1, 1, 0]).view(-1, 1)
XOR_weights_init(model)

loss_func = nn.L1Loss()
optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)

epochs = 2001
steps = X.size(0)
for i in range(epochs):
    for j in range(steps):
        data_point = np.random.randint(X.size(0))
        x_var = Variable(X[data_point], requires_grad=False)
        y_var = Variable(Y[data_point], requires_grad=False)

        optimizer.zero_grad()
        y_hat = model(x_var)
        loss = loss_func.forward(y_hat, y_var)
        loss.backward()
        optimizer.step()

    if i % 50 == 0:
        print("Epoch: {0}, Loss: {1}, ".format(i, loss.data.numpy()))
