import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import models
from data import heat, HeatFix
from utils import Optim
from utils import Heat
from utils import weight_init
from torchnet import meter
import os.path as osp
from typing import Callable
from torch import Tensor
from config import opt
import numpy as np
from numpy import pi
# import numba as nb



# -------------------------------------------------------------------------------------------------------------------------------------
# exact solution
# @nb.njit(fastmath = True)
# def kernel(x, y, t, m, n):
#     a = (2 - (2*((m + 1) % 2)))*(1 - np.cos(0.5*n*pi))
#     b = np.sin(0.5*m*pi*x)*np.sin(0.5*n*pi*y)
#     c = np.exp(-(np.power(pi, 2))*(np.power(m, 2) + np.power(n, 2))*t/36)
#     return a*b*c/(m*n)

# @nb.njit(fastmath = True)
# def Series_Sum(x, y, t, m, n):
#     res = 0.
#     for i in np.linspace(1, m, m):
#         for j in np.linspace(1, n, n):
#             res += kernel(x, y, t, i, j)
#             # print(res)
#     return 200*res/(pi**2)
# -------------------------------------------------------------------------------------------------------------------------------------


def train(**kwargs):
    # -------------------------------------------------------------------------------------------------------------------------------------
    # load the exact solution if exist
    opt._parse(kwargs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gridpath = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'exact_sol', opt.grid)
    exactpath = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'exact_sol', opt.exact)
    grid = torch.load(gridpath, map_location = device)
    exact = torch.load(exactpath, map_location = device)
    
    # fix gird
    # --------------------------------------------------------------------
    # datI =  grid[torch.logical_and(grid[:,0] != 0., grid[:,0] != 2),:]
    # datI = datI[torch.logical_and(datI[:,1] != 0., datI[:,1] != 2),:]
    
    # lrb = grid[torch.logical_or(grid[:,0] == 0., grid[:,0] == 2),:]
    # tbb = grid[torch.logical_or(grid[:,1] == 0., grid[:,1] == 2),:]
    # datB = torch.cat((lrb, tbb), dim = 0)
    # --------------------------------------------------------------------
    
    # grid = torch.load(gridpath, map_location = 'cpu')
    # grid = grid.numpy()
    # # timestamp = [2, 20, 40]
    timestamp = list(range(0, 102))
    # m, n = 100, 100
    # exact = []
    # for step in timestamp:
    #     tmp = [Series_Sum(i[0], i[1], step*0.0005, m, n) for i in grid]
    #     tmp = torch.tensor(tmp, device = device, dtype=torch.float32)
    #     exact.append(tmp.unsqueeze_(1))
    # grid = torch.tensor(grid, device = device, dtype=torch.float32)
    # -------------------------------------------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------------------------------------------
    # model configuration, modified the DATASET_MAP and LOSS_MAP according to your need
    DATASET_MAP = {'heat': heat}
    LOSS_MAP = {'heat':Heat}
    ACTIVATION_MAP = {'relu' : nn.ReLU(),
                    'tanh' : nn.Tanh(),
                    'sigmoid': nn.Sigmoid(),
                    'leakyrelu': nn.LeakyReLU()}
    keys = {'FClayer':opt.FClayer, 
            'num_blocks':opt.num_blocks,
            'activation':ACTIVATION_MAP[opt.act],
            'num_input':opt.num_input,
            'num_output':opt.num_oupt, 
            'num_node':opt.num_node}
    gendat = DATASET_MAP[opt.functional]
    losfunc = LOSS_MAP[opt.functional]
    # -------------------------------------------------------------------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------------------------------------------------------------------
    # model initialization
    model = getattr(models, opt.model)(**keys)
    model.to(device)
    # model.apply(weight_init)
    modelold = getattr(models, opt.model)(**keys)
    modelold.to(device)
    error = []
    datI = gendat(num = 2500, boundary = False, device = device)
    datB = gendat(num = 500, boundary = True, device = device)
    # datI = HeatFix(grid, boundary = False, device = device)
    # datB = HeatFix(grid, boundary = True, device = device)
    
    previous = []
    if opt.pretrain is None:
        with torch.no_grad():
            previous.append(model(datI))
            previous.append(model(datB))
        modelold.load_state_dict(model.state_dict())
    else:
        init_path = osp.join(osp.dirname(osp.realpath(__file__)), 'checkpoints', opt.pretrain)
        modelold.load_state_dict(torch.load(init_path))
        with torch.no_grad():
            previous.append(modelold(datI))
            previous.append(modelold(datB))
    # -------------------------------------------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------------------------------------------
    # train part
    for epoch in range(opt.max_epoch):
        # ---------------training setup in each time step---------------
        step = 0
        op = Optim(model.parameters(), opt)
        optimizer = op.optimizer
        scheduler = StepLR(optimizer, step_size=opt.step_size, gamma=opt.lr_decay)
        oldenergy = 1e-8
        # ---------------------------------------------------------------
        # --------------Optimization Loop at each time step--------------
        while True:
            optimizer.zero_grad()
            datI = gendat(num = 2500, boundary = False, device = device)
            datB = gendat(num = 500, boundary = True, device = device)
            
            # datI = HeatFix(grid, boundary = False, device = device)
            # datB = HeatFix(grid, boundary = True, device = device)
            with torch.no_grad():
                previous[0] = modelold(datI)
                previous[1] = modelold(datB)
            loss = losfunc(model, datI, datB, previous) 
            loss[0].backward()
            total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()).to(device) for p in model.parameters()]))
            nn.utils.clip_grad_norm_(model.parameters(),  1)
            optimizer.step()
            scheduler.step()
            step += 1      
            if total_norm < 1e-4 or step == 1200:
                break
            # if step == 2000:
            #     break
            # oldenergy = loss[1].item()
        if epoch in timestamp:
            # opt.lr = opt.lr * opt.lr_decay
            # error.append(abserr(model, grid, exact[timestamp.index(epoch + 1)]))
            print(abserr(model, grid, exact[timestamp.index(epoch + 1)]), '===============================================================')
            # print(abserr(model, grid, exact[timestamp.index(epoch + 1)]), '===============================================================')
            if epoch % 10 == 0:
                model.save(f'heat{epoch}.pt')
        modelold.load_state_dict(model.state_dict()) 
    print('=============================50====================')
    # torch.save(error, 'error103.pt')
            
    
    # -------------------------------------------------------------------------------------------------------------------------------------

@torch.no_grad()
def eval(model: Callable[..., Tensor], 
        grid: Tensor, 
        exact: Tensor):
    """
    Compute the relative L2 norm
    """
    model.eval()
    pred = model(grid)
    err  = torch.pow(torch.mean(torch.pow(pred - exact, 2))/torch.mean(torch.pow(exact, 2)), 0.5)
    model.train()
    return err

@torch.no_grad()
def abserr(model: Callable[..., Tensor], 
        grid: Tensor, 
        exact: Tensor):
    """
    Compute the relative L2 norm
    """
    model.eval()
    pred = model(grid)
    err = torch.mean(abs(pred - exact))
    # print(f"the mean abs error is {torch.mean(err)}")
    model.train()
    return err







def help():
    """
    Print out the help information： python file.py help
    """
    
    print("""
    usage : python file.py <function> [--args=value]
    <function> := train | make_plot | help
    Example: 
            python {0} train --lr=1e-5
            python {0} help

    Avaiable args: please refer to config.py""".format(__file__))

    # from inspect import getsource
    # source = (getsource(opt.__class__))
    # print(source)

if __name__=='__main__':
    import fire
    fire.Fire()










