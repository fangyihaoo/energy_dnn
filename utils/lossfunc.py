import torch

def criterion(model, dat_i, dat_b):

    g = dat_i.clone()
    g.requires_grad = True
    output_g = model(g)
    output_i = model(dat_i)
    output_b = model(dat_b)
    ux = torch.autograd.grad(outputs = output_g, inputs = g, grad_outputs = torch.ones_like(output_g), retain_graph=True, create_graph=True)[0]
    
    loss_i =  torch.mean(0.5 * torch.sum(torch.pow(ux, 2),dim=1)- output_i)
    loss_b = torch.mean(torch.pow(output_b,2))
    
    return loss_i + 500*loss_b