import torch
from torch import Tensor
from typing import Callable, List, Tuple
from numpy import pi



def PoiLoss(model: Callable[..., Tensor], 
            dat_i: Tensor, 
            dat_b: Tensor,
            previous: List[Tensor]) -> Tuple[Tensor,Tensor]:
    """
    Loss function for 2d Poisson equation
    -\laplacia u = 2sin(x)cos(y),    u \in \Omega
    u = 0,                           u \in \partial \Omega (0, pi) \times (-pi/2, pi/2)

    Args:
        model (Callable[..., Tensor]): Network 
        dat_i (Tensor): Interior point
        dat_b (Tensor): Boundary point
        previous (Tuple[Tensor, Tensor]): Result from previous time step model. interior point for index 0, boundary for index 1

    Returns:
        Tuple[Tensor,Tensor]: loss
    """

    f = (2*torch.sin(dat_i[:,0])*torch.cos(dat_i[:,1])).unsqueeze_(1)
    dat_i.requires_grad = True
    output_i = model(dat_i)
    output_b = model(dat_b)
    ux = torch.autograd.grad(outputs = output_i, inputs = dat_i, grad_outputs = torch.ones_like(output_i), retain_graph=True, create_graph=True)[0]
    # f = (2*torch.sin(dat_i[:,0])*torch.cos(dat_i[:,1])).unsqueeze_(1)
    loss_i =  torch.mean(0.5 * torch.sum(torch.pow(ux, 2),dim=1,keepdim=True)- f*output_i)
    loss_b = torch.mean(torch.pow(output_b,2))
    if previous:
        loss_p = 50*torch.mean(torch.pow(output_i - previous[0], 2))
        loss_p += 50*torch.mean(torch.pow(output_b - previous[1], 2))
    else:
        loss_p = 0

    return loss_i + 500*loss_b + loss_p, loss_i


def PoiHighLoss(model: Callable[..., Tensor], 
            dat: Tensor, 
            previous: List[Tensor]) -> Tensor:
    """
    Loss function for 2d Poisson equation
    -\laplacia u = pi^2 \sum_k=1^d cos(pi*x_k),    u \in \Omega
    u = 0,                           u \in \partial \Omega (0, 1) ^ d

    Args:
        model (Callable[..., Tensor]): Network 
        dat (Tensor): Interior point
        previous (Tensor): Result from previous time step model. 

    Returns:
        Tensor: loss
    """

    f = pi*pi*torch.sum(torch.cos(pi*dat), dim = 1, keepdim = True)
    dat.requires_grad = True
    output = model(dat)
    ux = torch.autograd.grad(outputs = output, inputs = dat, grad_outputs = torch.ones_like(output), retain_graph=True, create_graph=True)[0]
    loss =  torch.mean(0.5 * torch.sum(torch.pow(ux, 2),dim=1,keepdim=True)- f*output)
    loss_p = 50*torch.mean(torch.pow(output - previous, 2))
    return loss +  loss_p


def AllenCahn2dLoss(model: Callable[..., Tensor], 
                    dat_i: Tensor, 
                    dat_b: Tensor, 
                    previous: List[Tensor]) -> Tuple[Tensor,Tensor]:
    """
    Loss function for 2d Allen-Cahn type problem
    
    \int (1/2 D(\delta phi)^2 + 1/4(phi^2 - 1)^2)dx
    x \in (-1,1)\times(-1,1)

    Args:
        model (Callable[..., Tensor]): Network
        dat_i (Tensor): Interior point
        dat_b (Tensor): Boundary point
        previous (Tuple[Tensor, Tensor]): Result from previous time step model. interior point for index 0, boundary for index 1

    Returns:
        Tuple[Tensor,Tensor]: loss
    """

    dat_i.requires_grad = True
    output_i = model(dat_i)
    output_b = model(dat_b)
    ux = torch.autograd.grad(outputs = output_i, inputs = dat_i, grad_outputs = torch.ones_like(output_i), retain_graph=True, create_graph=True)[0]

    loss_i =  torch.mean(0.5 * torch.sum(torch.pow(ux, 2),dim=1,keepdim=True) + 25*torch.pow(torch.pow(output_i, 2) - 1, 2))
    loss_b = torch.mean(torch.pow((output_b[torch.logical_or(dat_b[:,0] == 1., dat_b[:,0] == -1),:]  + 1), 2))
    loss_b += torch.mean(torch.pow((output_b[torch.logical_or(dat_b[:,1] == 1., dat_b[:,1] == -1),:]  - 1), 2))
    loss_p = 100*torch.mean(torch.pow(output_i - previous[0], 2))
    loss_p += 100*torch.mean(torch.pow(output_b - previous[1], 2))
    
    return loss_i + 3000*loss_b + loss_p, loss_i


def AllenCahnW(model: Callable[..., Tensor], 
               dat_i: Tensor, 
               dat_b: Tensor, 
               previous: List[Tensor]) -> Tuple[Tensor,Tensor]:
    """
    \int 0.5*|\nabla \phi|^2 + 0.25*(\phi^2 - 1)^2/epislon^2 dx + W*(\int\phidx - A)^2
    r = 0.25
    A = (1 - pi*(r**2))*(-1) + pi*(r**2)
    W = 1000

    Args:
        model (Callable[..., Tensor]): Network
        dat_i (Tensor): Interior point
        dat_b (Tensor): Boundary point
        previous (Tuple[Tensor, Tensor]): Result from previous time step model. interior point for index 0, boundary for index 1

    Returns:
        Tuple[Tensor,Tensor]: loss
    """
    r = 0.25
    A = (1 - pi*(r**2))*(-1) + pi*(r**2)
    dat_i.requires_grad = True
    output_i = model(dat_i)
    output_b = model(dat_b)
    ux = torch.autograd.grad(outputs = output_i, inputs = dat_i, grad_outputs = torch.ones_like(output_i), retain_graph=True, create_graph=True)[0]

    loss_i =  torch.mean(0.5 * torch.sum(torch.pow(ux, 2),dim=1,keepdim=True) + 250*torch.pow(torch.pow(output_i, 2) - 1, 2)) 
    loss_b = torch.mean(torch.pow((output_b + 1), 2))
    loss_w = 1000*torch.pow((torch.mean(output_i) - A), 2)
    loss_p = 100*torch.mean(torch.pow(output_i - previous[0], 2))
    loss_p += 100*torch.mean(torch.pow(output_b - previous[1], 2))

    return loss_i + 500*loss_b + loss_w + loss_p, loss_i + loss_w


def AllenCahnLB(model: Callable[...,Tensor], 
                dat_i: Tensor, 
                dat_b: Tensor, 
                previous: List[Tensor]) -> Tuple[Tensor,Tensor]:
    """
    1/|\Omega|\int xi^2/2 (\laplacian Phi + phi)^2 + \tau/2 * \phi^2 - \gamma/6 * \phi^3 + 1/24 * \phi^4 dx
    (-1, 1)\times(-1, 1)

    Args:
        model (Callable[..., Tensor]): Network
        dat_i (Tensor): Interior point
        dat_b (Tensor): Boundary point
        previous (Tuple[Tensor, Tensor]): Result from previous time step model. interior point for index 0, boundary for index 1

    Returns:
        Tuple[Tensor,Tensor]: loss
    """
    dat_i.requires_grad = True
    output_i = model(dat_i)
    output_b = model(dat_b)
    ux = torch.autograd.grad(outputs = output_i, inputs = dat_i, grad_outputs = torch.ones_like(output_i), retain_graph=True, create_graph=True)[0]
    uxx = torch.autograd.grad(outputs = ux, inputs = dat_i, grad_outputs = torch.ones_like(ux), create_graph=True)[0]

    loss_i = 0.5*torch.pow(torch.sum(uxx, dim=1, keepdim=True) + output_i, 2)
    loss_i += 0.5*torch.pow(output_i, 2) + torch.pow(output_i, 4)/24
    loss_i = loss_i.mean()/4
    loss_v = torch.mean(output_i)
    loss_b = torch.mean(torch.pow((output_b  + 1), 2))
    loss_p = 100*torch.mean(torch.pow(output_i - previous[0], 2))
    loss_p += 100*torch.mean(torch.pow(output_b - previous[1], 2))

    return loss_i + 500*loss_v + 500*loss_b + loss_p, loss_i



def HeatPINN(model: Callable[..., Tensor], 
              dat_i: Tensor, 
              dat_b: Tensor, 
              dat_f: Tensor) -> Tensor:
    """The loss function for heat equation with PINN

    Args:
        model (Callable[..., Tensor]): Network 
        dat_i (Tensor): Initial points
        dat_b (Tensor): Boundary points
        dat_f (Tensor): Collocation points

    Returns:
        Tensor: loss
    """
    output_i = model(dat_i)
    output_b = model(dat_b)
    dat_f.requires_grad = True
    f = (2*torch.sin(dat_f[:,0])*torch.cos(dat_f[:,1])).unsqueeze_(1)
    output_f = model(dat_f)
    du = torch.autograd.grad(outputs = output_f, inputs = dat_f, grad_outputs = torch.ones_like(output_f), retain_graph=True, create_graph=True)[0]
    ut = du[:,2].unsqueeze_(1)
    ddu = torch.autograd.grad(outputs = du, inputs = dat_f, grad_outputs = torch.ones_like(du), create_graph=True)[0]
    lu = ddu[:,0:2]
    
    loss = torch.mean(torch.pow(output_i, 2))
    loss += 100*torch.mean(torch.pow(output_b, 2))
    loss += torch.mean(torch.pow(ut - torch.sum(lu, dim=1, keepdim=True) - f, 2))
    
    return loss


def PoissPINN(model: Callable[..., Tensor], 
              dat_i: Tensor, 
              dat_b: Tensor) -> Tensor:
    """The loss function for poisson equation with PINN

    Args:
        model (Callable[..., Tensor]): Network 
        dat_i (Tensor): Interior points
        dat_b (Tensor): Boundary points

    Returns:
        Tensor: loss
    """
    
    bd = lambda x, y : (torch.sin(x)*torch.cos(y)).unsqueeze_(1) # for the exact solution at boundary
    
    output_b = model(dat_b)
    f = (2*torch.sin(dat_i[:,0])*torch.cos(dat_i[:,1])).unsqueeze_(1)
    dat_i.requires_grad = True
    output_i = model(dat_i)
    du = torch.autograd.grad(outputs = output_i, inputs = dat_i, grad_outputs = torch.ones_like(output_i), retain_graph=True, create_graph=True)[0]
    ddu = torch.autograd.grad(outputs = du, inputs = dat_i, grad_outputs = torch.ones_like(du), retain_graph=True, create_graph=True)[0]
    loss = torch.mean(torch.pow(output_b - bd(dat_b[:,0], dat_b[:,1]), 2))
    loss += torch.mean(torch.pow(torch.sum(ddu, dim=1, keepdim=True) + f, 2))
    
    return loss
    
    

def PoissCyclePINN(model: Callable[..., Tensor], 
                   dat_i: Tensor, 
                   dat_b: Tensor) -> Tensor:
    """The loss function for poisson equation with PINN (cycle)
    
    -\nabla u = 1,   u = 1 - 0.25(x^2 + y^2)

    Args:
        model (Callable[..., Tensor]): Network 
        dat_i (Tensor): Interior points
        dat_b (Tensor): Boundary points

    Returns:
        Tensor: loss
    """
    
    bd = lambda x, y : (1 - 0.25*(x**2 + y**2)).unsqueeze_(1) # for the exact solution at boundary
    
    output_b = model(dat_b)
    dat_i.requires_grad = True
    output_i = model(dat_i)
    du = torch.autograd.grad(outputs = output_i, inputs = dat_i, grad_outputs = torch.ones_like(output_i), retain_graph=True, create_graph=True)[0]
    ddu = torch.autograd.grad(outputs = du, inputs = dat_i, grad_outputs = torch.ones_like(du), retain_graph=True, create_graph=True)[0]
    loss = torch.mean(torch.pow(output_b - bd(dat_b[:,0], dat_b[:,1]), 2))
    loss += torch.mean(torch.pow(torch.sum(ddu, dim=1, keepdim=True) + 1, 2))
    
    return loss


    
def PoiCycleLoss(model: Callable[..., Tensor], 
            dat_i: Tensor, 
            dat_b: Tensor,
            previous: List[Tensor]) -> Tuple[Tensor,Tensor]:
    """
    Loss function for 2d Poisson equation
    -\laplacia u = 1,    u \in \Omega
    u = 0,                           u \in \partial \Omega x^2 + y^2 = 1

    Args:
        model (Callable[..., Tensor]): Network 
        dat_i (Tensor): Interior point
        dat_b (Tensor): Boundary point
        previous (Tuple[Tensor, Tensor]): Result from previous time step model. interior point for index 0, boundary for index 1

    Returns:
        Tuple[Tensor,Tensor]: loss
    """
    bd = lambda x, y : (1 - 0.25*(x**2 + y**2)).unsqueeze_(1) # for the exact solution at boundary

    dat_i.requires_grad = True
    output_i = model(dat_i)
    output_b = model(dat_b)
    ux = torch.autograd.grad(outputs = output_i, inputs = dat_i, grad_outputs = torch.ones_like(output_i), retain_graph=True, create_graph=True)[0]
    loss_i =  torch.mean(0.5 * torch.sum(torch.pow(ux, 2),dim=1,keepdim=True) - output_i)
    loss_b = torch.mean(torch.pow(output_b - bd(dat_b[:,0], dat_b[:,1]),2))
    
    if previous:
        loss_p = 50*torch.mean(torch.pow(output_i - previous[0], 2))
        loss_p += 50*torch.mean(torch.pow(output_b - previous[1], 2))
    else:
        loss_p = 0

    return loss_i + 500*loss_b + loss_p, loss_i
    
    
def PFVB(model: Callable[...,Tensor], 
         dat_i: Tensor, 
         dat_b: Tensor, 
         previous: Tuple[Tensor,Tensor]) -> Tuple[Tensor,Tensor]:
    """[summary]

    Args:
        model (Callable[..., Tensor]): Network 
        dat_i (Tensor): Interior point
        dat_b (Tensor): Boundary point
        previous (Tuple[Tensor, Tensor]): Result from previous time step model. interior point for index 0, boundary for index 1

    Returns:
        Tuple[Tensor,Tensor]: loss
    """

    coef = 100 # coef  =  1/epsilon**2, where  epsilon = 0.1
    r = 0.25
    A = (pi*(r**2) - 1) + pi*(r**2)
    B = 1.

    dat_i.requires_grad = True
    output_i = model(dat_i)
    output_b = model(dat_b)
    ux = torch.autograd.grad(outputs = output_i, inputs = dat_i, grad_outputs = torch.ones_like(output_i), retain_graph=True, create_graph=True)[0]
    uxx = torch.autograd.grad(outputs = ux, inputs = dat_i, grad_outputs = torch.ones_like(output_i), create_graph=True)[0]

    loss_i = 0.5*torch.pow(coef*output_i*(output_i**2 - 1) - torch.sum(uxx, dim=1, keepdim=True), 2)
    loss_i += 1000*torch.pow(torch.mean(0.5 * torch.sum(torch.pow(ux, 2),dim=1,keepdim=True) + 0.25*coef*torch.pow(torch.pow(output_i, 2) - B, 2)) - 1, 2)
    loss_i += 1000*torch.pow((torch.mean(output_i) - A), 2)
    loss_p = 50*torch.mean(torch.pow(output_i - previous[0], 2))
    # loss_p += 50*torch.mean(torch.pow(output_b - previous[1], 2))
    loss_b = torch.mean(torch.pow((output_b  + 1), 2))
    
    return loss_i + 1000*loss_b + loss_p, loss_i

def Heat(model: Callable[...,Tensor], 
         dat_i: Tensor, 
         dat_b: Tensor, 
         previous: Tuple[Tensor,Tensor]) -> Tuple[Tensor,Tensor]:
    """
    2d Heat equation: u_t = \nabla u, x \in [0,2]^2
    u(x, t) = 0, x \in \delta\Omega
    u(x, t) = 50, if x_2 <= 1. u(x, t) = 0, if x_2 > 1.

    Args:
        model (Callable[...,Tensor]): [description]
        dat_i (Tensor): [description]
        dat_b (Tensor): [description]
        previous (Tuple[Tensor,Tensor]): [description]

    Returns:
        Tuple[Tensor,Tensor]: [description]
    """

    dat_i.requires_grad = True
    output_i = model(dat_i)
    output_b = model(dat_b)
    ux = torch.autograd.grad(outputs = output_i, inputs = dat_i, grad_outputs = torch.ones_like(output_i), retain_graph=True, create_graph=True)[0]
    loss_i =  torch.mean(0.5 * torch.sum(torch.pow(ux, 2),dim=1,keepdim=True)) * 4
    loss_b = torch.mean(torch.pow(output_b,2))
    
    loss_p = 50*torch.mean(torch.pow(output_i - previous[0], 2)) * 4
    loss_p += 50*torch.mean(torch.pow(output_b - previous[1], 2)) * 4

    return loss_i + 500*loss_b + loss_p, loss_i
