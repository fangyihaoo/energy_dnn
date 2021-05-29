import torch
from torch.utils.data import Dataset
from pyDOE import lhs
from numpy import pi
import numpy as np


class Poisson(Dataset):
    r"""
    Dataset class for interior points and boundary points of 2d poisson (0, pi)\times(-2/pi, 2/pi)

    Boundary:  uniform on each boundary

    Interior: latin square sampling

    Args:
        num (int): number of points need to be sample. For interior points, the output would be 4\times number
        boundary (bool): Boundary or Interior
        device (str): cpu or cuda

    """

    def __init__(self, num: int = 1000, boundary: bool = False, device='cpu'):  
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


def poisson(num: int = 1000, boundary: bool = False, device='cpu'):
    r"""
    Full batch version of 2d poisson (0, pi)\times(-2/pi, 2/pi)

    Boundary:  uniform on each boundary

    Interior: latin square sampling

    Args:
        num (int): number of points need to be sample. For interior points, the output would be 4\times number
        boundary (bool): Boundary or Interior
        device (str): cpu or gpu

    Return:
        Date coordinates tensor (N \times 2 or 4N \times 2)
    """
    if boundary:
        tb = torch.cat((torch.rand(num, 1)* pi, torch.tensor([pi/2]).repeat(num)[:,None]), dim=1)
        bb = torch.cat((torch.rand(num, 1)* pi, torch.tensor([-pi/2.]).repeat(num)[:,None]), dim=1)
        rb = torch.cat((torch.tensor([pi]).repeat(num)[:,None], torch.rand(num, 1)*pi - pi/2), dim=1)
        lb = torch.cat((torch.tensor([0.]).repeat(num)[:,None], torch.rand(num, 1)*pi - pi/2), dim=1)
        data = torch.cat((tb, bb, rb, lb), dim=0)
        data = data.to(device)
        return data
    else:
        lb = np.array([0., -pi/2])
        ran = np.array([pi, pi])
        data = torch.from_numpy(ran*lhs(2, num) + lb).float().to(device)        # generate the interior points
        return data



class AllenCahn(Dataset):
    '''
    Dataset class for interior points and boundary points of Allen-Cahn type problem (0, \infity)\times(0, 1)^2

    Boundary:  uniform on each boundary

    Interior: latin square sampling

    Args:
        num (int): number of points need to be sample. For interior points, the output would be 4\times number
        boundary (bool): Boundary or Interior
        device (str): cpu or cuda

    '''
    def __init__(self, num: int = 1000, boundary: bool = False, device='cpu'):  
        if boundary:
            tb = torch.cat((torch.rand(num, 1), torch.tensor([1.]).repeat(num)[:,None]), dim=1)
            bb = torch.cat((torch.rand(num, 1)* pi, torch.tensor([0.]).repeat(num)[:,None]), dim=1)
            rb = torch.cat((torch.tensor([1.]).repeat(num)[:,None], torch.rand(num, 1)), dim=1)
            lb = torch.cat((torch.tensor([0.]).repeat(num)[:,None], torch.rand(num, 1)), dim=1)
            self.data = torch.cat((tb, bb, rb, lb), dim=0)
            self.data = self.data.to(device) 

        else:
            self.data = torch.from_numpy(lhs(2, num)).float().to(device)        # generate the interior points
        
    def __getitem__(self, index):
        x = self.data[index]
        return x
    
    def __len__(self):
        return len(self.data)



def allencahn(num: int = 1024, boundary: bool = False, device='cpu'):
    '''
    Full batch version of of Allen-Cahn type problem (0, \infity)\times(0, 1)^2

    Boundary:  uniform on each boundary

    Interior: latin square sampling

    Args:
        num (int): number of points need to be sample. For interior points, the output would be 4\times number
        boundary (bool): Boundary or Interior
        device (str): cpu or gpu

    '''
    if boundary:
        tb = torch.cat((torch.rand(num, 1), torch.tensor([1.]).repeat(num)[:,None]), dim=1)
        bb = torch.cat((torch.rand(num, 1)* pi, torch.tensor([0.]).repeat(num)[:,None]), dim=1)
        rb = torch.cat((torch.tensor([1.]).repeat(num)[:,None], torch.rand(num, 1)), dim=1)
        lb = torch.cat((torch.tensor([0.]).repeat(num)[:,None], torch.rand(num, 1)), dim=1)
        data = torch.cat((tb, bb, rb, lb), dim=0)
        data = data.to(device) 
        return data
    else:
        data = torch.from_numpy(lhs(2, num)).float().to(device)        # generate the interior points
        return data