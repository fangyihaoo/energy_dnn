import torch
from torch import Tensor
from numpy import pi
from typing import Tuple



def mesh2d(num: int,
           xlim: Tuple[float, float],
           ylim: Tuple[float, float]) -> Tensor:
    """
    Generate meshgrid in square for 2d.

    Args:
        num (int): number of interval in one axis
        xlim (Tuple[float, float]): left and right boundary for x-axis
        ylim (Tuple[float, float]): lower and upper boundary for y-axis

    Returns:
        Tensor: grid location (num*num, 2)
    """

    x = torch.linspace(xlim[0], xlim[1], num)
    y = torch.linspace(ylim[0], ylim[1], num)
    X, Y = torch.meshgrid(x, y)
    Z = torch.cat((X.flatten()[:, None], Y.flatten()[:, None]), dim=1)

    return Z

def mesh2dcycle(num: int,
           xlim: Tuple[float, float],
           ylim: Tuple[float, float]) -> Tensor:
    """
    Generate meshgrid in square for 2d.

    Args:
        num (int): number of interval in one axis
        xlim (Tuple[float, float]): left and right boundary for x-axis
        ylim (Tuple[float, float]): lower and upper boundary for y-axis

    Returns:
        Tensor: grid location (num*num, 2)
    """

    x = torch.linspace(xlim[0], xlim[1], num)
    y = torch.linspace(ylim[0], ylim[1], num)
    X, Y = torch.meshgrid(x, y)
    Z = torch.cat((X.flatten()[:, None], Y.flatten()[:, None]), dim=1)
    ind = torch.sum(torch.pow(Z,2), 1) <= 1
    return Z[ind, :]

def poi2d(grid: Tensor) -> Tensor:
    """
    Generate exact solution according to the following 2D poisson equation in the meshgrid
    -\laplacia u = 2sin(x)cos(y),    u \in \Omega
    u = 0,              u \in \partial \Omega (0, pi) \times (-pi/2, pi/2)

    exact:  u = sin(x)cos(y)

    Args:
        grid (Tensor): location of the grid tensor (N,  2)

    Returns:
        Tensor: exact solution (N, 1)
    """

    return (torch.sin(grid[:,0])*torch.cos(grid[:,1])).unsqueeze_(1)



def poiss2dcyc(grid: Tensor) -> Tensor:
    """
    Generate exact solution according to the following 1D Allen-Cahn type energy functional
    exact:  u = 1 - 0.25*(x^2 + y^2)
    Args:
        grid (Tensor): location of the grid tensor (N, 2)

    Returns:
        Tensor: exact solution (N, 1)
    """

    return (1 - 0.25*(grid[:,0]**2 + grid[:,1]**2)).unsqueeze_(1)


def PoiHighExact(grid: Tensor) -> Tensor:
    """
    Generate exact solution according to the following high dimension poisson equation

    Exact: u = \sum_{k = 1}^{d} cos(\pi * x_k)

    Args:
        grid (Tensor): location of the grid tensor (N, d)

    Returns:
        Tensor: exact solution (N, 1)
    """
    return torch.sum(torch.cos(pi*grid), dim = 1, keepdim = True)

def mesh3dsphere(
           M: int=99,
           N: int=100) -> Tensor:
    """
    Generate 3d meshgrid on sphere.
    https://stackoverflow.com/questions/4081898/procedurally-generate-a-sphere-mesh
    (x, y, z) = (sin(Pi * m/M) cos(2Pi * n/N), sin(Pi * m/M) sin(2Pi * n/N), cos(Pi * m/M))
    Args:
        m: number of latitudes.
        n: number of longitudes.
    Returns:
        Tensor: grid location ((M+1)*N, 2)
    """
    r = 1
    m = torch.arange(0,M+1)
    n = torch.arange(0,N)
    x =  r*torch.einsum('i,j->ij',torch.sin(pi*m/M),torch.cos(2*pi*n/N)).flatten()
    y =  r*torch.einsum('i,j->ij',torch.sin(pi*m/M),torch.sin(2*pi*n/N)).flatten()
    z = r*torch.cos(pi*m/M).unsqueeze_(1).repeat(1,N).flatten()
    data = data = torch.cat((x.unsqueeze_(1),y.unsqueeze_(1),z.unsqueeze_(1)),dim=1)
    return data

def poiss3dsphere(grid: Tensor) -> Tensor:
    """
    \laplace_{\Gamma} u = 18x1x2x3, u \on S^2.
    Generate exact solution : u = x1x2x3
    Args:
        grid (Tensor): location of the grid tensor (N, 3)
    Returns:
        Tensor: exact solution (N, 1)
    """

    return (grid[:,0]*grid[:,1]*grid[:,2]).unsqueeze_(1)

def poiss3dsphere2(grid: Tensor) -> Tensor:
    """
    \laplace_{\Gamma} u = 18x1x2x3, u \on S^2.
    Generate exact solution : u = x1sin(x2)+x3
    Args:
        grid (Tensor): location of the grid tensor (N, 3)
    Returns:
        Tensor: exact solution (N, 1)
    """

    return (grid[:,0]*torch.sin(grid[:,1])+grid[:,2]).unsqueeze_(1)





if __name__ == '__main__':

    """
    generate the mesh grid on (0., pi) times (-pi/2, pi/2)
    and the corresponding exact solution on the grid
    """
    # Z = mesh2d(101, (0., pi), (-pi/2, pi/2))                  # poisson 2d
    # exact = poi2d(Z)
    # torch.save(Z, '../data/exact_sol/poiss2dgrid.pt')
    # torch.save(exact, '../data/exact_sol/poiss2dexact.pt')

    # Z = mesh2dcycle(101, (-1., 1.), (-1., 1.))                  # poisson 2d cycle
    # exact = poiss2dcyc(Z)
    # torch.save(Z, '../data/exact_sol/poiss2dcyclegrid.pt')
    # torch.save(exact, '../data/exact_sol/poiss2dcycleexact.pt')

    # Z = mesh2d(101, (0., 2.), (0., 2.))                  # heat 2d
    # exact = poiss2dcyc(Z)
    # torch.save(Z, '../data/exact_sol/heatgrid.pt')
    # torch.save(exact, '../data/exact_sol/poiss2dcycleexact.pt')

    Z = mesh3dsphere()             # poisson 3d sphere
    exact = poiss3dsphere(Z)
    torch.save(Z, '../data/exact_sol/poiss3dspheregrid.pt')
    torch.save(exact, '../data/exact_sol/poiss3dsphereexact.pt')
