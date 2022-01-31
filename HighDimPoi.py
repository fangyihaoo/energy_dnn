import torch
import torch.nn as nn
import models
from data import PoiHighGrid
from utils import Optim
from utils import PoiHighLoss
from utils import PoiHighExact
from utils import weight_init
from utils import L2_Reg
# import os.path as osp
from typing import Callable
from torch import Tensor
from config import opt


def train(**kwargs):
    # -------------------------------------------------------------------------------------------------------------------------------------
    # load the setting
    opt._parse(kwargs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # -------------------------------------------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------------------------------------------
    # model configuration, modified the DATASET_MAP and LOSS_MAP according to your need
    DATASET_MAP = {'poi': PoiHighGrid}
    LOSS_MAP = {'poi':PoiHighLoss}
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
    model.apply(weight_init)
    modelold = getattr(models, opt.model)(**keys)
    modelold.to(device)
    previous = 0
    modelold.load_state_dict(model.state_dict())
    # -------------------------------------------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------------------------------------------
    # model optimizer and recorder
    timestamp = [20*i  for i in range(1, 10)]
    MinError = float('inf')
    # -------------------------------------------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------------------------------------------
    # train part
    for epoch in range(opt.max_epoch):
        # ---------------training setup in each time step---------------
        step = 0
        op = Optim(model.parameters(), opt)
        optimizer = op.optimizer
        # ---------------------------------------------------------------
        # --------------Optimization Loop at each time step--------------
        while True:
            optimizer.zero_grad()
            datI = gendat(num = 100000, d = opt.dimension, device = device)
            with torch.no_grad():
                previous = modelold(datI)
            weight = L2_Reg(model, modelold)
            loss = losfunc(model, datI, previous) + opt.lamda*weight
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),  1)
            optimizer.step()
            step += 1      
            if step == opt.step_size:
                break
        err = eval(model, datI, PoiHighExact(datI))
        if epoch in timestamp:
            opt.lr = opt.lr * opt.lr_decay
        if epoch % 10 == 0:
            if err < MinError: 
                model.save('HighPoi.pt')
                MinError = err
        print(f'The epoch is {epoch}, The error is {err}')
        modelold.load_state_dict(model.state_dict())

    
    # -------------------------------------------------------------------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------------------------------------------------------------------
    """
    l2 relative error
    """
    # error = []
    # for _ in range(100):
    #     datI = gendat(num = 2000, d = opt.dimension, device = device)
    #     error.append(eval(model, datI, PoiHighExact(datI)))
    
    # print('the mean is :', torch.mean(torch.tensor(error)))
    # print('the sd is : ',  torch.std(torch.tensor(error), unbiased=True))
    
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