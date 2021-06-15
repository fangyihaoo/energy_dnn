import torch
import torch.nn as nn
import models
from data import poisson, allencahn
from utils import Optim
from utils import PoiLoss,  PoiCycleLoss
from utils import weight_init
from torchnet import meter
import os.path as osp
from typing import Callable
from torch import Tensor
from config import opt


def train(**kwargs):
    # -------------------------------------------------------------------------------------------------------------------------------------
    # load the exact solution if exist
    opt._parse(kwargs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exactpath = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'exact_sol', opt.exact)
    gridpath = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'exact_sol', opt.grid)
    grid = torch.load(gridpath, map_location = device)
    exact = torch.load(exactpath, map_location = device)
    # -------------------------------------------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------------------------------------------
    # model configuration, modified the DATASET_MAP and LOSS_MAP according to your need
    DATASET_MAP = {'poi': poisson,
                   'poissoncycle': allencahn}
    LOSS_MAP = {'poi':PoiLoss,
                'poissoncycle': PoiCycleLoss}
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
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model.to(device)
    # model.apply(weight_init)
    modelold = getattr(models, opt.model)(**keys)
    modelold.to(device)
    datI = gendat(num = 1000, boundary = False, device = device)
    datB = gendat(num = 250, boundary = True, device = device)
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
    # model optimizer and recorder
    op = Optim(model.parameters(), opt)
    optimizer = op.optimizer
    timestamp = [5000*i  for i in range(1, 10)]
    error = []
    # -------------------------------------------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------------------------------------------
    # train part
    for epoch in range(opt.max_epoch + 1):
        # ---------------training setup in each time step---------------
        step = 0
        op = Optim(model.parameters(), opt)
        optimizer = op.optimizer
        oldenergy = 1e-8
        # ---------------------------------------------------------------
        # --------------Optimization Loop at each time step--------------
        while True:
            optimizer.zero_grad()
            datI = gendat(num = 1000, boundary = False, device = device)
            datB = gendat(num = 100, boundary = True, device = device)
            with torch.no_grad():
                previous[0] = modelold(datI)
                previous[1] = modelold(datB)
            loss = losfunc(model, datI, datB, previous) 
            loss[0].backward()
            nn.utils.clip_grad_norm_(model.parameters(),  10)
            optimizer.step()
            step += 1        
            if abs((loss[1].item() - oldenergy)/oldenergy) < 1e-5 or step == 5000:
                break
            oldenergy = loss[1].item()
        if epoch in timestamp:
            opt.lr = opt.lr * opt.lr_decay
        err = eval(model, grid, exact)
        error.append(err)
        if epoch % 1000 == 0:
            print(f'The epoch is {epoch}, The error is {err}')
        modelold.load_state_dict(model.state_dict())
    error = torch.FloatTensor(error)
    torch.save(error, osp.join(osp.dirname(osp.realpath(__file__)), 'log', 'decay', opt.functional + 'poissourmethod.pt'))
    
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



        # datI = DATASET_MAP[opt.functional](num = 2000, boundary = False, device = device)
        # datB = DATASET_MAP[opt.functional](num = 25, boundary = True, device = device)
        # datI_loader = DataLoader(datI, 200, shuffle=True) # make sure that the dataloders are the same len for datI and datB
        # datB_loader = DataLoader(datB, 10, shuffle=True)




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