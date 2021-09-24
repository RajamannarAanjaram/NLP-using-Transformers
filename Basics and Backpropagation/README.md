<br>
<h1 align="center"> Basics and Backpropagation
<br>
    
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
[![Open Source? Yes!](https://badgen.net/badge/Open%20Source%20%3F/Yes%21/blue?icon=github)](https://github.com/RajamannarAanjaram/badges/)
[![Awesome Badges](https://img.shields.io/badge/badges-awesome-green.svg)](https://github.com/RajamannarAanjaram/badges)
    <br>
[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![ForTheBadge built-with-love](http://ForTheBadge.com/images/badges/built-with-love.svg)](https://GitHub.com/RajamannarAanjaram/)
</h1>

<h3 align="left">
<strong>Basic question</strong>
</h3>

1. What is a neuron in neural network<br>
<ul>
<li>The neuron in <strong>Neural Network</strong> is conceptually derived from biological neurons</li>
<li>Each neurons can take single/multiple inputs and produces only one output</li>
<li>To find the output of the neuron, first we take the weighted sum of all the inputs, weighted by the weights of the connections from the inputs to the neuron. We add a bias term to this sum. This weighted sum is sometimes called the <strong>activation</strong></li>
</ul>

2. What is the use of the learning rate<br>
<ul>
<li>learning rate is a hyperparameter which determines the step size at each iteration while moving toward a minimum of a loss function</li>
<li>learning rate determines how big a step is taken in that direction</li>
<li>A too high learning rate will make the learning jump over minima but a too low learning rate will either take too long to converge or get stuck in an undesirable local minimum</li>
</ul>

3. How are weights initialized<br>
<ul>
<li>Weights are a square matrix which are randomly initialized</li>
<li>Weights can also be assigned with sets of 1’s and 0’s</li>
<li>If a matrix if full of zero’s except on the center column, we can that the kernel filters the straight line</li>
<li>A value 0 in kernel means it will not consider the pixel in convolution</li>
</ul>

4. What is "loss" in a neural network<br>
<ul>
    <li>Loss helps us to understand how much the predicted value differ from actual value</li>
    <li>Function used to calculate the loss is called as <strong>Loss function</strong></li>
    <li>Loss function is a method of evaluating how well your algorithm models your dataset</li>
    <li>If your predictions are totally off, your loss function will output a higher number</li>
    <li>If they’re pretty good, it’ll output a lower number</li>
    <li>We should train the model in such a way that the loss is minimal</li>
</ul>

Model architecture 

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

Parameter used in the model
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Linear-1                    [-1, 5]              15
            Linear-2                    [-1, 4]              24
            Linear-3                    [-1, 1]               5
================================================================
Total params: 44
Trainable params: 44
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.00
Params size (MB): 0.00
Estimated Total Size (MB): 0.00
----------------------------------------------------------------
```

Log for 2001 epochs is linked [here](./log.md)
    
    
    
    
