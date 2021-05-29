import torch
from numpy import pi



def PoiLoss(model, dat_i, dat_b):

    r"""
    Loss function for 2d Poisson equation
    -\laplacia u = 2sin(x)cos(y),    u \in \Omega
    u = 0,                           u \in \partial \Omega (0, pi) \times (-pi/2, pi/2)
    """

    dat_i.requires_grad = True
    output_i = model(dat_i)
    output_b = model(dat_b)
    ux = torch.autograd.grad(outputs = output_i, inputs = dat_i, grad_outputs = torch.ones_like(output_i), retain_graph=True, create_graph=True)[0]
    f = (2*torch.sin(dat_i[:,0])*torch.cos(dat_i[:,1])).unsqueeze_(1)
    loss_i =  torch.mean(0.5 * torch.sum(torch.pow(ux, 2),dim=1,keepdim=True)- f*output_i)
    loss_b = torch.mean(torch.pow(output_b,2))

    return loss_i + 500*loss_b


def AllenCahnLoss(model, dat_i, dat_b):

    r"""
    Loss function for 2d Allen-Cahn type problem
    
    \int (1/2 D(\delta phi)^2 + 1/4(phi^2 - 1)^2)dx
    x \in (0,1)\times(0,1)
    """

    dat_i.requires_grad = True
    output_i = model(dat_i)
    output_b = model(dat_b)
    ux = torch.autograd.grad(outputs = output_i, inputs = g, grad_outputs = torch.ones_like(output_i), retain_graph=True, create_graph=True)[0]

    loss_i =  torch.mean(0.5 * torch.sum(torch.pow(ux, 2),dim=1,keepdim=True) + 0.25*torch.pow(torch.pow(output_i, 2) - 1., 2))
    
    loss_b = torch.mean(torch.pow((output_b[torch.logical_or(dat_b[:,0] == 1., dat_b[:,0] == 0),:]  - 1), 2))
    loss_b += torch.mean(torch.pow((output_b[torch.logical_or(dat_b[:,1] == 1., dat_b[:,1] == 0),:]  + 1), 2))

    return loss_i + 500*loss_b


