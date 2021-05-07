import torch



def criterion(model, dat_i, dat_b):
    '''
    loss function for 2d Poisson equation
    \laplacia u = 1,    u \in \Omega
    u = 0,              u \in \partial \Omega (-1, 1) \times (-1, 1)

    '''

    g = dat_i.clone()
    g.requires_grad = True
    output_g = model(g)
    output_i = model(dat_i)
    output_b = model(dat_b)
    ux = torch.autograd.grad(outputs = output_g, inputs = g, grad_outputs = torch.ones_like(output_g), retain_graph=True, create_graph=True)[0]
    f = 2*torch.sin(dat_i[:,0])*torch.cos(dat_i[:,1])
    
    loss_i =  torch.mean(0.5 * torch.sum(torch.pow(ux, 2),dim=1)- 2*f*output_i)
    loss_b = torch.mean(torch.pow(output_b,2))
    
    return loss_i + 500*loss_b