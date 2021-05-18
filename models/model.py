import torch
import torch.nn as nn
from .basic_module import BasicModule
from typing import Type


class ResBlock(nn.Module):
    '''
    Residule block
    '''

    def __init__(self, num_node, num_fc, activate = nn.Tanh()):
        super(ResBlock, self).__init__()
        self.activate = activate
        self.linears_list = [nn.Linear(num_node, num_node) for i in range(num_fc)]
        self.acti_list = [self.activate for i in range(num_fc)]
        self.block = nn.Sequential(*[item for pair in zip(self.linears_list, self.acti_list) for item in pair])


    def forward(self, x):
        residual = x
        out = self.block(x)
        out = out + residual                  # dont' put inplace addition here if inline activation
        # out = self.activate(out)
        return out
        
        
class ResNet(BasicModule):
    '''
    Residule network
    '''

    def __init__(self, 
        FClayer: int = 2,                                           # number of fully-connected layers in one residual block
        num_blocks: int = 4,                                        # number of residual blocks
        activation = nn.Tanh(),                                     # activation function
        num_input: int = 2,                                         # dimension of input, in this case is 2 
        num_node: int = 10                                          # number of nodes in one fully-connected layer
    ) -> None:
        super(ResNet, self).__init__()
        self.num_blocks = num_blocks
        self.activation = activation
        self.input = nn.Linear(num_input, num_node)     
        for i in range(self.num_blocks):
            setattr(self,f'ResiB{i}',ResBlock(num_node, FClayer, self.activation))
        self.output = nn.Linear(num_node, 1)

        
    def _forward_impl(self, x):
        
        x = self.input(x)
        # x = self.activation(x)
        for i in range(self.num_blocks):
            x = getattr(self, f'ResiB{i}')(x)
        x = self.output(x)        
        return x
    
    def forward(self, x):
        return self._forward_impl(x)



class Pinn(BasicModule):
    '''
    Fully connected network
    '''

    def __init__(self, 
        FClayer: int = 2,                                           # number of fully-connected layers
        activation = nn.Tanh(),                                     # activation function
        num_layer: int = 5,                                          # number of layers
        num_input: int = 2,                                         # dimension of input, in this case is 2 
        num_node: int = 20,                                          # number of nodes in one fully-connected layer
        num_oupt: int = 1                                           # dimension of output, in this case is 1
    ) -> None:
        super(Pinn, self).__init__()

        self.input = nn.Linear(num_input, num_node)
        self.act = activation
        self.output = nn.Linear(num_node, num_oupt)

        'Fully connected blocks'     
        self.linears_list = [nn.Linear(num_node, num_node) for i in range(num_layer)]
        self.acti_list = [self.act for i in range(num_layer)]
        self.block = nn.Sequential(*[item for pair in zip(self.linears_list, self.acti_list)for item in pair])

    
    def forward(self, x):

        x = self.input(x)
        x = self.block(x)
        x = self.output(x)
        return x
