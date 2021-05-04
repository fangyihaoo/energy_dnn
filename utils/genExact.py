import torch
from torch import Tensor


def mesh2d(num: int) -> Tensor:
    '''
    Generate meshgrid in square

    input:  
    num: number of interval in one axis

    output: grid location in Tensor form with dim = (num*num, 2)
    '''

    x = torch.linspace(-1, 1, num)
    y = torch.linspace(-1, 1, num)
    X, Y = torch.meshgrid(x, y)
    Z = torch.cat((Y.flatten()[:, None], Y.T.flatten()[:, None]), dim=1)

    return Z

def poi2d(grid: Tensor) -> Tensor:
    '''
    Generate exact solution according to the following 2D poisson equation in the meshgrid
    \laplacia u = 1,    u \in \Omega
    u = 0,              u \in \partial \Omega (-1, 1) \times (-1, 1)

    exact:  u = 1/4(x^2 + y^2 - 2)

    input:  
    grid: location of the grid tensor

    output: exact solution in Tensor form with dim = (num*num)
    '''

    return -(torch.sum(grid**2, dim = 1) - 2)/4





if __name__ == '__main__':
    Z = mesh2d(1001)
    exact = poi2d(Z)
    torch.save(Z, '../data/exact_sol/poiss2dgrid.pt')
    torch.save(exact, '../data/exact_sol/poiss2dexact.pt')