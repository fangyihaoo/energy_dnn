import torch
from torch import Tensor
import torch.nn as nn
from .basic_module import BasicModule
from typing import Callable


class ResBlock(nn.Module):
    '''
    Residule block
    '''

    def __init__(self, 
    num_node: int, 
    num_fc: int, 
    activate: Callable[..., Tensor] = nn.Tanh()
    ) -> None:

        super(ResBlock, self).__init__()
        self.activate = activate
        self.linears_list = [nn.Linear(num_node, num_node) for i in range(num_fc)]
        self.acti_list = [self.activate for i in range(num_fc)]
        self.norm = [nn.BatchNorm1d(num_features=num_node) for i in range(num_fc)]
        block = [item for x in zip(linears_list, norm, acti_list) for item in x]
        self.block = nn.Sequential(*block)
        # self.block = nn.Sequential(*[item for pair in zip(self.linears_list, self.acti_list) for item in pair])

    def forward(self, x):
        residual = x
        out = self.block(x)
        out = out + residual                  # dont' put inplace addition here if inline activation
        return out
        
        
class ResNet(BasicModule):
    r"""
        Residule network
    """

    def __init__(self, 
        FClayer: int = 2,                                           # number of fully-connected layers in one residual block
        num_blocks: int = 3,                                        # number of residual blocks
        activation: Callable[..., Tensor] = nn.Tanh(),              # activation function
        num_input: int = 2,                                         # dimension of input, in this case is 2 
        num_node: int = 10,                                          # number of nodes in one fully-connected layer
        num_oupt: int = 1,                                          # dimension of output
        **kwargs
    ) -> None:

        super(ResNet, self).__init__()
        self.num_blocks = num_blocks
        self.activation = activation
        self.input = nn.Linear(num_input, num_node)     
        for i in range(self.num_blocks):
            setattr(self,f'ResiB{i}',ResBlock(num_node, FClayer, self.activation))
        self.output = nn.Linear(num_node, num_oupt)

    def _forward_impl(self, x):
        x = self.input(x)
        for i in range(self.num_blocks):
            x = getattr(self, f'ResiB{i}')(x)
        x = self.output(x)        
        return x
    
    def forward(self, x):
        return self._forward_impl(x)



class FullNet(BasicModule):
    r"""
        MLP
    """

    def __init__(self, 
        FClayer: int = 5,                                                    # number of fully-connected hidden layers
        activation: Callable[..., Tensor] = nn.Tanh(),                       # activation function
        num_input: int = 2,                                                  # dimension of input, in this case is 2 
        num_node: int = 20,                                                  # number of nodes in one fully-connected hidden layer
        num_oupt: int = 1,                                                   # dimension of output
        **kwargs
    ) -> None:

        super(FullNet, self).__init__()
        self.input = nn.Linear(num_input, num_node)
        self.act = activation
        self.output = nn.Linear(num_node, num_oupt)

        'Fully connected blocks'     
        self.linears_list = [nn.Linear(num_node, num_node) for i in range(FClayer)]
        self.acti_list = [self.act for i in range(FClayer)]
        self.block = nn.Sequential(*[item for pair in zip(self.linears_list, self.acti_list)for item in pair])

    def forward(self, x):
        x = self.input(x)
        x = self.block(x)
        x = self.output(x)
        return x
