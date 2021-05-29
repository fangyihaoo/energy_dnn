import torch
from torch import Tensor
from numpy import pi
from typing import Tuple



def mesh2d(num: int, 
            xlim: Tuple[float, float], 
            ylim: Tuple[float, float]) -> Tensor:
    '''
    Generate meshgrid in square for 2d.

    input:  
        num: number of interval in one axis
        xlim: left and right boundary for x-axis
        ylim: lower and upper boundary for y-axis

    output: grid location (num*num, 2)
    '''

    x = torch.linspace(xlim[0], xlim[1], num)
    y = torch.linspace(ylim[0], ylim[1], num)
    X, Y = torch.meshgrid(x, y)
    Z = torch.cat((X.flatten()[:, None], Y.flatten()[:, None]), dim=1)

    return Z



def poi2d(grid: Tensor) -> Tensor:
    '''
    Generate exact solution according to the following 2D poisson equation in the meshgrid
    -\laplacia u = 2sin(x)cos(y),    u \in \Omega
    u = 0,              u \in \partial \Omega (0, pi) \times (-pi/2, pi/2)

    exact:  u = sin(x)cos(y)

    input:  
        grid: location of the grid tensor (N,  2)

    output: 
        exact solution (N, 1)
    '''

    return (torch.sin(grid[:,0])*torch.cos(grid[:,1])).unsqueeze_(1)



def allen2d(grid: Tensor) -> Tensor:
    '''
    Generate exact solution according to the following 2D Allen-Cahn type energy functional
    

    exact:  

    input:  
        grid: location of the grid tensor (N, 2)

    output: 
        exact solution (N, 1)
    '''

    pass









if __name__ == '__main__':
    Z = mesh2d(201, (0., pi), (-pi/2, pi/2))                  # poisson 2d
    exact = poi2d(Z)
    torch.save(Z, '../data/exact_sol/poiss2dgrid.pt')
    torch.save(exact, '../data/exact_sol/poiss2dexact.pt')