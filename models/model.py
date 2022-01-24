import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
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
        # self.norm = [nn.BatchNorm1d(num_features=num_node) for i in range(num_fc)]
        # block = [item for x in zip(self.linears_list, self.norm, self.acti_list) for item in x]
        # self.block = nn.Sequential(*block)
        self.block = nn.Sequential(*[item for pair in zip(self.linears_list, self.acti_list) for item in pair])

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


"""
The Augmented-ICNN code is directly copy from https://github.com/CW-Huang/CP-Flow

"""

_scaling_min = 0.001

class ActNorm(torch.nn.Module):
    """ ActNorm layer with data-dependant init."""

    def __init__(self, num_features, logscale_factor=1., scale=1., learn_scale=True):
        super(ActNorm, self).__init__()
        self.initialized = False
        self.num_features = num_features

        self.register_parameter('b', nn.Parameter(torch.zeros(1, num_features, 1), requires_grad=True))
        self.learn_scale = learn_scale
        if learn_scale:
            self.logscale_factor = logscale_factor
            self.scale = scale
            self.register_parameter('logs', nn.Parameter(torch.zeros(1, num_features, 1), requires_grad=True))

    def forward_transform(self, x, logdet=0):
        input_shape = x.size()
        x = x.view(input_shape[0], input_shape[1], -1)

        if not self.initialized:
            self.initialized = True

            def unsqueeze(x):
                return x.unsqueeze(0).unsqueeze(-1).detach()

            # Compute the mean and variance
            sum_size = x.size(0) * x.size(-1)
            b = -torch.sum(x, dim=(0, -1)) / sum_size
            self.b.data.copy_(unsqueeze(b).data)

            if self.learn_scale:
                var = unsqueeze(torch.sum((x + unsqueeze(b)) ** 2, dim=(0, -1)) / sum_size)
                logs = torch.log(self.scale / (torch.sqrt(var) + 1e-6)) / self.logscale_factor
                self.logs.data.copy_(logs.data)

        b = self.b
        output = x + b

        if self.learn_scale:
            logs = self.logs * self.logscale_factor
            scale = torch.exp(logs) + _scaling_min
            output = output * scale
            dlogdet = torch.sum(torch.log(scale)) * x.size(-1)  # c x h

            return output.view(input_shape), logdet + dlogdet
        else:
            return output.view(input_shape), logdet

    def reverse(self, y, **kwargs):
        assert self.initialized
        input_shape = y.size()
        y = y.view(input_shape[0], input_shape[1], -1)
        logs = self.logs * self.logscale_factor
        b = self.b
        scale = torch.exp(logs) + _scaling_min
        x = y / scale - b

        return x.view(input_shape)

    def extra_repr(self):
        return f"{self.num_features}"


class ActNormNoLogdet(ActNorm):

    def forward(self, x):
        return super(ActNormNoLogdet, self).forward_transform(x)[0]



def symm_softplus(x, softplus_=torch.nn.functional.softplus):
    return softplus_(x) - 0.5 * x


def softplus(x):
    return nn.functional.softplus(x)


def gaussian_softplus(x):
    z = np.sqrt(np.pi / 2)
    return (z * x * torch.erf(x / np.sqrt(2)) + torch.exp(-x**2 / 2) + z * x) / (2*z)


def gaussian_softplus2(x):
    z = np.sqrt(np.pi / 2)
    return (z * x * torch.erf(x / np.sqrt(2)) + torch.exp(-x**2 / 2) + z * x) / z


def laplace_softplus(x):
    return torch.relu(x) + torch.exp(-torch.abs(x)) / 2


def cauchy_softplus(x):
    # (Pi y + 2 y ArcTan[y] - Log[1 + y ^ 2]) / (2 Pi)
    pi = np.pi
    return (x * pi - torch.log(x**2 + 1) + 2 * x * torch.atan(x)) / (2*pi)


def activation_shifting(activation):
    def shifted_activation(x):
        return activation(x) - activation(torch.zeros_like(x))
    return shifted_activation


def get_softplus(softplus_type='softplus', zero_softplus=False):
    if softplus_type == 'softplus':
        act = nn.functional.softplus
    elif softplus_type == 'gaussian_softplus':
        act = gaussian_softplus
    elif softplus_type == 'gaussian_softplus2':
        act = gaussian_softplus2
    elif softplus_type == 'laplace_softplus':
        act = gaussian_softplus
    elif softplus_type == 'cauchy_softplus':
        act = cauchy_softplus
    else:
        raise NotImplementedError(f'softplus type {softplus_type} not supported.')
    if zero_softplus:
        act = activation_shifting(act)
    return act


class Softplus(nn.Module):
    def __init__(self, softplus_type='softplus', zero_softplus=False):
        super(Softplus, self).__init__()
        self.softplus_type = softplus_type
        self.zero_softplus = zero_softplus

    def forward(self, x):
        return get_softplus(self.softplus_type, self.zero_softplus)(x)


class SymmSoftplus(torch.nn.Module):

    def forward(self, x):
        return symm_softplus(x)


class PosLinear(torch.nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        gain = 1 / x.size(1)
        return nn.functional.linear(x, torch.nn.functional.softplus(self.weight), self.bias) * gain
    

class AugICNN(torch.nn.Module):
    def __init__(self, dim=2, dimh=16, num_hidden_layers=2, symm_act_first=False,
                 softplus_type='softplus', zero_softplus=False):
        super(AugICNN, self).__init__()

        self.act = Softplus(softplus_type=softplus_type, zero_softplus=zero_softplus)
        self.symm_act_first = symm_act_first

        Wzs = list()
        Wzs.append(nn.Linear(dim, dimh))
        for _ in range(num_hidden_layers - 1):
            Wzs.append(PosLinear(dimh, dimh // 2, bias=True))
        Wzs.append(PosLinear(dimh, 1, bias=False))
        self.Wzs = torch.nn.ModuleList(Wzs)

        Wxs = list()
        for _ in range(num_hidden_layers - 1):
            Wxs.append(nn.Linear(dim, dimh // 2))
        Wxs.append(nn.Linear(dim, 1, bias=False))
        self.Wxs = torch.nn.ModuleList(Wxs)

        Wx2s = list()
        for _ in range(num_hidden_layers - 1):
            Wx2s.append(nn.Linear(dim, dimh // 2))
        self.Wx2s = torch.nn.ModuleList(Wx2s)

        actnorms = list()
        for _ in range(num_hidden_layers - 1):
            actnorms.append(ActNormNoLogdet(dimh // 2))
        actnorms.append(ActNormNoLogdet(1))
        actnorms[-1].b.requires_grad_(False)
        self.actnorms = torch.nn.ModuleList(actnorms)

    def forward(self, x):
        if self.symm_act_first:
            z = symm_softplus(self.Wzs[0](x), self.act)
        else:
            z = self.act(self.Wzs[0](x))
        for Wz, Wx, Wx2, actnorm in zip(self.Wzs[1:-1], self.Wxs[:-1], self.Wx2s[:], self.actnorms[:-1]):
            z = self.act(actnorm(Wz(z) + Wx(x)))
            aug = Wx2(x)
            aug = symm_softplus(aug, self.act) if self.symm_act_first else self.act(aug)
            z = torch.cat([z, aug], 1)
        return self.actnorms[-1](self.Wzs[-1](z) + self.Wxs[-1](x))