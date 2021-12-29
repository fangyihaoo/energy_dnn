import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import models
from data import heat
from utils import Optim
from utils import Heat
from utils import weight_init
import os.path as osp
from typing import Callable
from torch import Tensor
from config import opt



def train(**kwargs):
    # -------------------------------------------------------------------------------------------------------------------------------------
    # load the exact solution if exist
    opt._parse(kwargs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gridpath = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'exact_sol', opt.grid)
    exactpath = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'exact_sol', opt.exact)
    grid = torch.load(gridpath, map_location = device)
    exact = torch.load(exactpath, map_location = device)
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
    model.apply(weight_init)
    modelold = getattr(models, opt.model)(**keys)
    modelold.to(device)
    timestamp = set([1, 20, 60, 100])
    datI = gendat(num = 2500, boundary = False, device = device)
    datB = gendat(num = 500, boundary = True, device = device)

    
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
    for epoch in range(opt.max_epoch+1):
        # ---------------training setup in each time step---------------
        step = 0
        op = Optim(model.parameters(), opt)
        optimizer = op.optimizer
        scheduler = StepLR(optimizer, step_size=opt.step_size, gamma=opt.lr_decay)
        # ---------------------------------------------------------------
        # --------------Optimization Loop at each time step--------------
        while True:
            optimizer.zero_grad()
            datI = gendat(num = 2500, boundary = False, device = device)
            datB = gendat(num = 500, boundary = True, device = device)
            
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
        print(abserr(model, grid, exact[epoch]), step)
        if epoch in timestamp:
            model.save(f'heat{epoch}.pt')
        modelold.load_state_dict(model.state_dict()) 
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










