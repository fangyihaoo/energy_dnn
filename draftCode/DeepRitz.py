import torch
from torch import Tensor
import torch.nn as nn
from pyDOE import lhs
from torch.utils.data import Dataset, DataLoader
from typing import Type, Any, Callable, Union, List, Optional



class ResBlock(nn.Module):
    def __init__(self, num_node, num_fc, activate = nn.Tanh()):
        super(ResBlock, self).__init__()
        self.act = activate
        self.linears_list = [nn.Linear(num_node, num_node) for i in range(num_fc)]
        self.acti_list = [self.act for i in range(num_fc)]
        self.block = nn.Sequential(*[item for pair in zip(self.linears_list, self.acti_list) for item in pair])
        
        'Xavier Normal Initialization'        
        for m in self.block:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data,  gain=1.0)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out = out + residual                  # dont' put inplace addition here
#         out = self.act(out)
        return out
        
        
class ResNet(nn.Module):
    def __init__(self, 
        block: Type[ResBlock], 
        FClayer: int = 2,                                           # number of fully-connected layers in one residual block
        num_blocks: int = 4,                                        # number of residual blocks
        activation = nn.Tanh(),                                     # activation function
        num_node: int = 10                                          # number of nodes in one fully-connected layer
    ) -> None:
        super(ResNet, self).__init__()
        self.num_blocks = num_blocks
        self.act = activation
        self.input = nn.Linear(2, num_node)
        nn.init.xavier_normal_(self.input.weight.data,  gain=1.0)
        for i in range(self.num_blocks):
            setattr(self,f'ResiB{i}',block(num_node, FClayer, self.act))
        self.output = nn.Linear(num_node, 1)
        nn.init.xavier_normal_(self.output.weight.data,  gain=1.0)
        
    def  _forward_impl(self, x):
        x = self.input(x)
        x = self.act(x)
        for i in range(self.num_blocks):
            x = getattr(self, f'ResiB{i}')(x)
        x = self.output(x)        
        return x
    
    def forward(self, x):
        return self._forward_impl(x)
    

    
# 'mini-batch version'
# def GenDat(device, N_i = 128, N_b = 33):
#     """

#     Implements 

#     -\delta u(x) = 1 two dimention
#             u(x) = 0 boundary

#     INPUT:
#         N_i -- number of interior data points 
#         N_b -- N_b*5 number of boundary data points 
   
#     OUTPUT:
#     x_i, x_b -- size (N_i,3) interior points, size (N_b,3) boundary points, third dimension is the label, 0 means collocation, 1 means boundary

#     """

#     x_i = 2*lhs(2, N_i) - 1
#     x_i = torch.from_numpy(x_i).float()
#     x_i = torch.hstack((x_i, torch.zeros(N_i)[:,None]))
    

#     zeorb = torch.cat((torch.rand(N_b, 1), torch.tensor([0.]).repeat(N_b)[:,None]), dim=1)
#     upb   = torch.cat((torch.rand(N_b, 1)* 2 - 1, torch.tensor([1.]).repeat(N_b)[:,None]), dim=1)
#     lowb  = torch.cat((torch.rand(N_b, 1)* 2 - 1, torch.tensor([-1.]).repeat(N_b)[:,None]), dim=1)
#     rb    = torch.cat((torch.tensor([1.]).repeat(N_b)[:,None], torch.rand(N_b, 1)* 2 - 1), dim=1)
#     lb    = torch.cat((torch.tensor([-1.]).repeat(N_b)[:,None], torch.rand(N_b, 1)* 2 - 1), dim=1)
#     x_b   = torch.cat((zeorb, upb, lowb, rb, lb), dim=0)
#     x_b   = torch.hstack((x_b, torch.ones(N_b*5)[:,None]))
    
#     x_i = x_i.to(device)
#     x_b = x_b.to(device)
    
#     return x_i, x_b
    
    
    
# 'mini-batch version'
# def Loss(model, dat):
#     ix = dat[:,2] == 0
    
#     dat_i = dat[ix,:2]
#     dat_b = dat[~ix,:2]
    
#     g = dat_i.clone()
#     g.requires_grad = True
#     output_g = model(g)
#     output_i = model(dat_i)
#     output_b = model(dat_b)
#     ux = torch.autograd.grad(outputs = output_g, inputs = g, grad_outputs = torch.ones_like(output_g), retain_graph=True, create_graph=True)[0]
    
#     loss_r =  torch.mean(0.5 * torch.sum(torch.pow(ux, 2),dim=1)- output_i)
#     loss_b = torch.mean(torch.pow(output_b,2))
  
#     if dat_i.shape[0] == 0:
#         return loss_b
    
#     if dat_b.shape[0] == 0:
#         return loss_r
    
#     return loss_r + 500*loss_b


class Poisson(Dataset):
    def __init__(self, x_i, x_b):       
        self.data = torch.vstack((x_i,x_b))
        
    def __getitem__(self, index):
        x = self.data[index]
        return x
    
    def __len__(self):
        return len(self.data)

        
'full-batch version'
def GenDat(device, N_i = 128, N_b = 33):
    """

    Implements 

    -\delta u(x) = 1 two dimention
            u(x) = 0 boundary

    INPUT:
        N_i -- number of interior data points 
        N_b -- N_b*5 number of boundary data points 
   
    OUTPUT:
    x_i, x_b -- size (N_i,3) interior points, size (N_b,3) boundary points, third dimension is the label, 0 means collocation, 1 means boundary

    """

    x_i = 2*lhs(2, N_i) - 1
    x_i = torch.from_numpy(x_i).float()   
    
    zeorb = torch.cat((torch.rand(N_b, 1), torch.tensor([0.]).repeat(N_b)[:,None]), dim=1)
    upb   = torch.cat((torch.rand(N_b, 1)* 2 - 1, torch.tensor([1.]).repeat(N_b)[:,None]), dim=1)
    lowb  = torch.cat((torch.rand(N_b, 1)* 2 - 1, torch.tensor([-1.]).repeat(N_b)[:,None]), dim=1)
    rb    = torch.cat((torch.tensor([1.]).repeat(N_b)[:,None], torch.rand(N_b, 1)* 2 - 1), dim=1)
    lb    = torch.cat((torch.tensor([-1.]).repeat(N_b)[:,None], torch.rand(N_b, 1)* 2 - 1), dim=1)
    x_b   = torch.cat((zeorb, upb, lowb, rb, lb), dim=0)
   
    x_i = x_i.to(device)
    x_b = x_b.to(device)
    
    return x_i, x_b

def Loss(model, dat_i, dat_b):

    g = dat_i.clone()
    g.requires_grad = True
    output_g = model(g)
    output_i = model(dat_i)
    output_b = model(dat_b)
    ux = torch.autograd.grad(outputs = output_g, inputs = g, grad_outputs = torch.ones_like(output_g), retain_graph=True, create_graph=True)[0]
    
    loss_r =  torch.mean(0.5 * torch.sum(torch.pow(ux, 2),dim=1)- output_i)
    loss_b = torch.mean(torch.pow(output_b,2))
    
    return loss_r + 500*loss_b