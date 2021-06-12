import torch
import torch.nn as nn
# from torch.optim.lr_scheduler import StepLR
import models
from data import poisspinn
from utils import Optim
from utils import PoissPINN
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
    DATASET_MAP = {'poi': poisspinn}
    LOSS_MAP = {'poi':PoissPINN}
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
    # -------------------------------------------------------------------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------------------------------------------------------------------
    # model initialization
    model = getattr(models, opt.model)(**keys)
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model.to(device)
    model.apply(weight_init)

    # -------------------------------------------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------------------------------------------
    # model optimizer and recorder
    op = Optim(model.parameters(), opt)
    optimizer = op.optimizer
    # scheduler = StepLR(optimizer, step_size= opt.step_size, gamma = opt.lr_decay)
    previous_err = 10000
    best_epoch = 0
    # -------------------------------------------------------------------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------------------------------------------------------------------
    # train part
    for epoch in range(opt.max_epoch + 1):
        optimizer.zero_grad()
        datF = DATASET_MAP[opt.functional](num = 5000, data_type = 'collocation', device = device)
        datI = DATASET_MAP[opt.functional](num = 500, data_type = 'initial', device = device)
        datB = DATASET_MAP[opt.functional](num = 100, data_type = 'boundary', device = device)
        loss = LOSS_MAP[opt.functional](model, datI, datB, datF) 
        loss.backward()
        optimizer.step()
        # scheduler.step()
        if epoch % 500 == 0:
            err = eval(model, grid, exact)
            print(f'Epoch: {epoch:05d}  Loss: {loss.item():.5f}   Error: {err.item():.5f}')
    model.save(f'poissonpinn{epoch}.pt')

    # -------------------------------------------------------------------------------------------------------------------------------------

@torch.no_grad()
def eval(model: Callable[..., Tensor], 
        grid: Tensor, 
        exact: Tensor) -> Tensor:
    """
    Compute the relative L2 norm

    Args:
        model (Callable[..., Tensor]): Network
        grid (Tensor): grid of exact solution
        exact (Tensor): exact solution

    Returns:
        Tensor: Relative L2 norm
    """
    model.eval()
    pred = model(grid)
    err  = torch.pow(torch.mean(torch.pow(pred - exact, 2))/torch.mean(torch.pow(exact, 2)), 0.5)
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


if __name__=='__main__':
    import fire
    fire.Fire()

