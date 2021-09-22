from torch.autograd import Variable
import numpy as np
import torch.optim as optim
import torch.nn as nn


def train_XOR(model,X,Y, epoch=300, lr=0.02, momentum=0.9):
    
    loss_func = nn.L1Loss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    epoch = 2001
    steps = X.size(0)
    for i in range(epoch):
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