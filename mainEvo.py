import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import models
from data import poisson, allencahn
from utils import Optim
from utils import PoiLoss, AllenCahn2dLoss, AllenCahnW, AllenCahnLB
from utils import weight_init
# from torch.utils.data import DataLoader
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
    # exactpath1 = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'exact_sol', 'allen2dexact1.pt')
    # exactpath2 = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'exact_sol', 'allen2dexact2.pt')
    # gridpath = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'exact_sol', 'allen2dgrid.pt')
    # grid = torch.load(gridpath, map_location = device)
    # exact1 = torch.load(exactpath1, map_location = device)
    # exact2 = torch.load(exactpath2, map_location = device)
    # -------------------------------------------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------------------------------------------
    # model configuration, modified the DATASET_MAP and LOSS_MAP according to your need
    DATASET_MAP = {'poi': poisson,
                    'allenw': allencahn}
    LOSS_MAP = {'poi':PoiLoss,
                'allen': AllenCahn2dLoss,
                'allenw': AllenCahnW,
                'allenlb': AllenCahnLB}
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
    modelold = getattr(models, opt.model)(**keys)
    modelold.to(device)
    datI = DATASET_MAP[opt.functional](num = 2500, boundary = False, device = device)
    datB = DATASET_MAP[opt.functional](num = 100, boundary = True, device = device)
    previous = []
    if opt.pretrain is None:
        ini_dat = torch.zeros_like(datI)
        b_dat = torch.zeros_like(datB)
        with torch.no_grad():
            previous.append(model(ini_dat))
            previous.append(model(b_dat))
        modelold.load_state_dict(model.state_dict())
    else:
        init_path = osp.join(osp.dirname(osp.realpath(__file__)), 'checkpoints', opt.pretrain)
        modelold.load_state_dict(torch.load(init_path))
        with torch.no_grad():
            previous.append(model(datI))
            previous.append(model(datB))
    # -------------------------------------------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------------------------------------------
    # model optimizer and recorder
    op = Optim(model.parameters(), opt)
    optimizer = op.optimizer
    loss_meter = meter.AverageValueMeter()
    timestamp = [10, 40, 300]
    energylist = []
    log_path = osp.join(osp.dirname(osp.realpath(__file__)), 'log', 'evolution',  'energys.pt')
    # -------------------------------------------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------------------------------------------
    # train part
    for epoch in range(opt.max_epoch + 1):
        # ---------------training setup in each time step---------------
        loss_meter.reset()
        energys = []
        step = 0
        op = Optim(model.parameters(), opt)
        optimizer = op.optimizer
        scheduler = StepLR(optimizer, step_size=opt.step_size, gamma=opt.lr_decay)
        oldenergy = 1e-8
        # ---------------------------------------------------------------
        # --------------Optimization Loop at each time step--------------
        while True:
            optimizer.zero_grad()
            datI = DATASET_MAP[opt.functional](num = 2500, boundary = False, device = device)
            datB = DATASET_MAP[opt.functional](num = 100, boundary = True, device = device)
            with torch.no_grad():
                previous[0] = modelold(datI)
                previous[1] = modelold(datB)
            loss = LOSS_MAP[opt.functional](model, datI, datB, previous) 
            loss[0].backward()
            nn.utils.clip_grad_norm_(model.parameters(),  1)
            optimizer.step()
            scheduler.step()
            loss_meter.add(loss[1].item())
            step += 1
            if epoch in timestamp:
                energys.append(loss[1].item())            
            if abs((loss[1].item() - oldenergy)/oldenergy) < 1e-4 or step == 5000:
                break
            oldenergy = loss[1].item()
        if epoch in timestamp:
            energylist.append(energys)
            model.save(f'allencahn{epoch}.pt')
        modelold.load_state_dict(model.state_dict())
        # if epoch % 10 == 0:
        #     log = 'Epoch: {:05d}, Loss: {:.5f}'
        #     print(log.format(epoch, torch.tensor(loss_meter.value()[0])))
    torch.save(gradstotal, log_path)

    # -------------------------------------------------------------------------------------------------------------------------------------

@torch.no_grad()
def eval(model: Callable[..., Tensor], 
        grid: Tensor, 
        exact: Tensor):
    r"""
    Compute the relative L2 norm
    """
    model.eval()
    pred = model(grid)
    err  = torch.pow(torch.mean(torch.pow(pred - exact, 2))/torch.mean(torch.pow(exact, 2)), 0.5)
    model.train()
    return err


        
        # if epoch % 100 == 0:
        #     test_err1 = eval(model, grid, exact1)
        #     test_err2 = eval(model, grid, exact2)
        #     if test_err1 < test_err2:
        #         if test_err1 < previous_err:
        #             previous_err = test_err1
        #             best_epoch = epoch
        #             flag = True
        #         else:
        #             pass
        #     else:
        #         if test_err2 < previous_err:
        #             previous_err = test_err2
        #             best_epoch = epoch
        #             flag = False
        #         else:
        #             pass
        #     log = 'Epoch: {:05d}, Loss: {:.5f}, Test: {:.5f}, Best Epoch: {:05d}, Which: {}'
        #     print(log.format(epoch, torch.abs(torch.tensor(loss_meter.value()[0])), previous_err.item(), best_epoch, flag))
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

