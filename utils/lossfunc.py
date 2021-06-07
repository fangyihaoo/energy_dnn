import torch
from numpy import pi



def PoiLoss(model, dat_i, dat_b):

    r"""
    Loss function for 2d Poisson equation
    -\laplacia u = 2sin(x)cos(y),    u \in \Omega
    u = 0,                           u \in \partial \Omega (0, pi) \times (-pi/2, pi/2)
    """
    f = (2*torch.sin(dat_i[:,0])*torch.cos(dat_i[:,1])).unsqueeze_(1)
    dat_i.requires_grad = True
    output_i = model(dat_i)
    output_b = model(dat_b)
    ux = torch.autograd.grad(outputs = output_i, inputs = dat_i, grad_outputs = torch.ones_like(output_i), retain_graph=True, create_graph=True)[0]
    # f = (2*torch.sin(dat_i[:,0])*torch.cos(dat_i[:,1])).unsqueeze_(1)
    loss_i =  torch.mean(0.5 * torch.sum(torch.pow(ux, 2),dim=1,keepdim=True)- f*output_i)
    loss_b = torch.mean(torch.pow(output_b,2))

    return loss_i + 500*loss_b


def AllenCahn2dLoss(model, dat_i, dat_b):

    r"""
    Loss function for 2d Allen-Cahn type problem
    
    \int (1/2 D(\delta phi)^2 + 1/4(phi^2 - 1)^2)dx
    x \in (0,1)\times(0,1)
    """

    dat_i.requires_grad = True
    output_i = model(dat_i)
    output_b = model(dat_b)
    ux = torch.autograd.grad(outputs = output_i, inputs = dat_i, grad_outputs = torch.ones_like(output_i), retain_graph=True, create_graph=True)[0]

    loss_i =  torch.mean(0.5 * torch.sum(torch.pow(ux, 2),dim=1,keepdim=True) + 25*torch.pow(torch.pow(output_i, 2) - 1, 2))
    loss_b = torch.mean(torch.pow((output_b[torch.logical_or(dat_b[:,0] == 1., dat_b[:,0] == 0),:]  + 1), 2))
    loss_b += torch.mean(torch.pow((output_b[torch.logical_or(dat_b[:,1] == 1., dat_b[:,1] == 0),:]  - 1), 2))

    return loss_i + 500*loss_b

def AllenCahnW(model, dat_i, dat_b, previous):

    r"""
    \int 0.5*|\nabla \phi|^2 + 0.25*(\phi^2 - 1)^2/epislon^2 dx + W*(\int\phidx - A)^2
    r = 0.25
    A = (1 - pi*(r**2))*(-1) + pi*(r**2)
    W = 1000
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
    loss_p = 1000*torch.mean(torch.pow(output_i - previous[0], 2))
    loss_p += 1000*torch.mean(torch.pow(output_b - previous[1], 2))

    return loss_i + 500*loss_b + loss_w + loss_p, loss_i + loss_w


def AllenCahnLB(model, dat_i, dat_b):
    r"""
    1/|\Omega|\int xi^2/2 (\laplacian Phi + phi)^2 + \tau/2 * \phi^2 - \gamma/6 * \phi^3 + 1/24 * \phi^4 dx
    (-1, 1)\times(-1, 1)
    """
    dat_i.requires_grad = True
    output_i = model(dat_i)
    output_b = model(dat_b)
    ux = torch.autograd.grad(outputs = output_i, inputs = dat_i, grad_outputs = torch.ones_like(output_i), retain_graph=True, create_graph=True)[0]
    uxx = torch.autograd.grad(outputs = ux, inputs = dat_i, grad_outputs = torch.ones(ux.size()), create_graph=True)[0]

    loss_i = 0.5*torch.pow(torch.sum(uxx, dim=1, keepdim=True) + output_i, 2)
    loss_i += 0.5*torch.pow(output_i, 2) + torch.pow(output_i, 4)/4
    loss_i = loss_i.mean()/4
    loss_i += torch.mean(output_i)
    loss_b = torch.mean(torch.pow((output_b  + 1), 2))

    return loss_i + 500*loss_b