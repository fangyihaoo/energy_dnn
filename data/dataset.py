import torch
from torch.utils.data import Dataset
from pyDOE import lhs
from numpy import pi
import numpy as np


# class Poisson(Dataset):
#     '''
#     Dataset for interior points and boundary points of 2d poisson (-1, 1)\times(-1, 1)

#     Boundary:  uniform on each boundary

#     Interior: latin square sampling

#     '''

#     def __init__(self, num = 1000, boundary = False, device='cpu'):  
#         if boundary:
#             tb = torch.cat((torch.rand(num, 1)* 2 - 1, torch.tensor([1.]).repeat(num)[:,None]), dim=1)
#             bb = torch.cat((torch.rand(num, 1)* 2 - 1, torch.tensor([-1.]).repeat(num)[:,None]), dim=1)
#             rb = torch.cat((torch.tensor([1.]).repeat(num)[:,None], torch.rand(num, 1)* 2 - 1), dim=1)
#             lb = torch.cat((torch.tensor([-1.]).repeat(num)[:,None], torch.rand(num, 1)* 2 - 1), dim=1)
#             self.data = torch.cat((tb, bb, rb, lb), dim=0)
#             self.data = self.data.to(device) 

#         else:  
#             self.data = torch.from_numpy(2*lhs(2, num) - 1).float().to(device)        # generate the interior points
        
#     def __getitem__(self, index):
#         x = self.data[index]
#         return x
    
#     def __len__(self):
#         return len(self.data)


class Poisson(Dataset):
    '''
    Dataset for interior points and boundary points of 2d poisson (0, pi)\times(-2/pi, 2/pi)

    Boundary:  uniform on each boundary

    Interior: latin square sampling

    '''

    def __init__(self, num = 1000, boundary = False, device='cpu'):  
        if boundary:
            tb = torch.cat((torch.rand(num, 1)* pi, torch.tensor([pi/2]).repeat(num)[:,None]), dim=1)
            bb = torch.cat((torch.rand(num, 1)* pi, torch.tensor([-pi/2.]).repeat(num)[:,None]), dim=1)
            rb = torch.cat((torch.tensor([pi]).repeat(num)[:,None], torch.rand(num, 1)*pi - pi/2), dim=1)
            lb = torch.cat((torch.tensor([0.]).repeat(num)[:,None], torch.rand(num, 1)*pi - pi/2), dim=1)
            self.data = torch.cat((tb, bb, rb, lb), dim=0)
            self.data = self.data.to(device) 

        else:
            lb = np.array([0., -pi/2])
            ran = np.array([pi, pi])
            self.data = torch.from_numpy(ran*lhs(2, num) + lb).float().to(device)        # generate the interior points
        
    def __getitem__(self, index):
        x = self.data[index]
        return x
    
    def __len__(self):
        return len(self.data)