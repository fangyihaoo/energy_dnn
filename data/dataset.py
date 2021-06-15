import torch
from torch import Tensor
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

    def __init__(self, num: int = 1000, boundary: bool = False, device: str ='cpu'):  
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


def poisson(num: int = 1000, 
            boundary: bool = False, 
            device: str ='cpu') -> Tensor:
    """
    2d poisson (0, pi)\times(-2/pi, 2/pi)
    Boundary:  uniform on each boundary
    Interior: latin square sampling

    Args:
        num (int): number of points need to be sample. For interior points, the output would be 4\times number
        boundary (bool): Boundary or Interior
        device (str): cpu or gpu

    Returns:
        Tensor: Date coordinates tensor (N \times 2 or 4N \times 2)
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
    """
    Dataset class for interior points and boundary points of Allen-Cahn type problem (0, \infity)\times(0, 1)^2
    Boundary:  uniform on each boundary
    Interior: latin square sampling

    Args:
        num (int): number of points need to be sample. For interior points, the output would be 4\times number
        boundary (bool): Boundary or Interior
        device (str): cpu or cuda
    """
    
    def __init__(self, num: int = 1000, boundary: bool = False, device: str ='cpu'):  
        if boundary:
            tb = torch.cat((torch.rand(num, 1)*2 - 1, torch.tensor([1.]).repeat(num)[:,None]), dim=1)
            bb = torch.cat((torch.rand(num, 1)*2 - 1, torch.tensor([-1.]).repeat(num)[:,None]), dim=1)
            rb = torch.cat((torch.tensor([1.]).repeat(num)[:,None], torch.rand(num, 1)*2 - 1), dim=1)
            lb = torch.cat((torch.tensor([-1.]).repeat(num)[:,None], torch.rand(num, 1)*2 - 1), dim=1)
            self.data = torch.cat((tb, bb, rb, lb), dim=0)
            self.data = self.data.to(device) 

        else:
            self.data = torch.from_numpy(lhs(2, num)*2 - 1).float().to(device)        # generate the interior points
            #self.data = torch.rand((num,2)).to(device)
        
    def __getitem__(self, index):
        x = self.data[index]
        return x
    
    def __len__(self):
        return len(self.data)



def allencahn(num: int = 1000, 
              boundary: bool = False, 
              device: str ='cpu') -> Tensor:
    """
    Allen-Cahn type problem (0, \infity)\times(0, 1)^2
    Boundary:  uniform on each boundary
    Interior: latin square sampling

    Args:
        num (int): number of points need to be sample. For interior points, the output would be 4\times number
        boundary (bool): Boundary or Interior
        device (str): cpu or gpu

    Returns:
        Tensor: Date coordinates tensor (N \times 2 or 4N \times 2)
    """
    
    if boundary:
        tb = torch.cat((torch.rand(num, 1)*2 - 1, torch.tensor([1.]).repeat(num)[:,None]), dim=1)
        bb = torch.cat((torch.rand(num, 1)*2 - 1, torch.tensor([-1.]).repeat(num)[:,None]), dim=1)
        rb = torch.cat((torch.tensor([1.]).repeat(num)[:,None], torch.rand(num, 1)*2 - 1), dim=1)
        lb = torch.cat((torch.tensor([-1.]).repeat(num)[:,None], torch.rand(num, 1)*2 - 1), dim=1)
        data = torch.cat((tb, bb, rb, lb), dim=0)
        data = data.to(device) 
        return data
    else:
        data = torch.from_numpy(lhs(2, num)*2 - 1).float().to(device)        # generate the interior points
        return data
    
    
def heatpinn(num: int = 1000, 
              data_type: str = 'boundary', 
              device: str = 'cpu') ->Tensor:
    r""" 
    2d poisson (0, pi)\times(-2/pi, 2/pi)\times(0,2) for PINN
    Boundary:  uniform on each boundary
    Interior: latin square sampling

    Args:
        num (int, optional): number of data points. Defaults to 1000.
        data_type (str, optional): boundary condition. Defaults to boundary.
        device (str, optional): 'cuda' or 'cpu'. Defaults to 'cpu'.

    Returns:
        Tensor: (num, 3) dimension Tensor
    """
    
    if data_type == 'boundary':
        tb = torch.cat((torch.rand(num, 1)* pi, torch.tensor([pi/2]).repeat(num)[:,None]), dim=1)
        bb = torch.cat((torch.rand(num, 1)* pi, torch.tensor([-pi/2.]).repeat(num)[:,None]), dim=1)
        rb = torch.cat((torch.tensor([pi]).repeat(num)[:,None], torch.rand(num, 1)*pi - pi/2), dim=1)
        lb = torch.cat((torch.tensor([0.]).repeat(num)[:,None], torch.rand(num, 1)*pi - pi/2), dim=1)
        data = torch.cat((tb, bb, rb, lb), dim=0)
        data = torch.cat((data, torch.rand(data.shape[0], 1)*2), dim = 1)
        return data.to(device)
    
    elif data_type == 'initial':
        lb = np.array([0., -pi/2])
        ran = np.array([pi, pi])
        data = torch.from_numpy(ran*lhs(2, num) + lb).float()        # generate the interior points
        data = torch.cat((data, torch.tensor([0]).repeat(data.shape[0])[:,None]), dim = 1)
        return data.to(device)
    
    else:
        lb = np.array([0., -pi/2])
        ran = np.array([pi, pi])
        data = torch.from_numpy(ran*lhs(2, num) + lb).float()       # generate the collocation points
        data = torch.cat((data, torch.rand(data.shape[0], 1)*2), dim = 1)
        return data.to(device)





# def poissoncycle(num: int = 1000, 
#                  boundary: bool = False,
#                  device: str = 'cpu') -> Tensor:
#     """
#     Poisson equation for -\laplacian u = 1
#     in (-1,1) \times (-1, 1), x^2 + y^2 <= 1

#     Args:
#         num (int, optional): number of data points. Defaults to 1000.
#         data_type (str, optional): boundary condition. Defaults to boundary.
#         device (str, optional): 'cuda' or 'cpu'. Defaults to 'cpu'.

#     Returns:
#         Tensor: (num, 2) dimension Tensor
#     """
#     theta = torch.rand(num)*2*pi
#     if boundary:
#         data = torch.cat((torch.cos(theta).unsqueeze_(1), torch.sin(theta).unsqueeze_(1) ), dim = 1)
#         return data.to(device)
#     else:
#         r = torch.rand(num)
#         data = torch.cat(((r*torch.cos(theta)).unsqueeze_(1), (r*torch.sin(theta)).unsqueeze_(1)), dim = 1)
#         return data.to(device)