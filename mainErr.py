import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR 
import models
from data import Poisson, AllenCahn, allencahn
from utils import Optim
from utils import PoiLoss, AllenCahn2dLoss
from utils import weight_init
from utils import plot
from torch.utils.data import DataLoader
from torchnet import meter
import os.path as osp
from typing import Callable
from torch import Tensor
from config import opt




def train(**kwargs):

    # setup
    opt._parse(kwargs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exactpath1 = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'exact_sol', 'allen2dexact1.pt')
    exactpath2 = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'exact_sol', 'allen2dexact2.pt')
    gridpath = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'exact_sol', 'allen2dgrid.pt')
    grid = torch.load(gridpath, map_location = device)
    exact1 = torch.load(exactpath1, map_location = device)
    exact2 = torch.load(exactpath2, map_location = device)
    flag = True

    # configure model
    DATASET_MAP = {'poi':Poisson,
                    'allen':AllenCahn}
    LOSS_MAP = {'poi':PoiLoss,
                'allen': AllenCahn2dLoss}
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
    
    model = getattr(models, opt.model)(**keys)
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model.to(device)
    model.apply(weight_init)

    op = Optim(model.parameters(), opt)
    optimizer = op.optimizer
    scheduler = StepLR(optimizer, step_size=opt.step_size, gamma=opt.lr_decay)
    loss_meter = meter.AverageValueMeter()

    # regularization the initial w0 value
    if opt.tau != 0:
        w0 = [0 for p in model.parameters()] 
    previous_err = torch.tensor(10000)
    best_epoch = 0

    for epoch in range(opt.max_epoch + 1):
        loss_meter.reset()
        # datI = DATASET_MAP[opt.functional](num = 2000, boundary = False, device = device)
        # datB = DATASET_MAP[opt.functional](num = 25, boundary = True, device = device)
        # datI_loader = DataLoader(datI, 200, shuffle=True) # make sure that the dataloders are the same len for datI and datB
        # datB_loader = DataLoader(datB, 10, shuffle=True)

        datI = allencahn(num = 2000, boundary = False, device = device)
        datB = allencahn(num = 25, boundary = True, device = device)
        # for data in zip(datI_loader, datB_loader):
        optimizer.zero_grad()
        # loss = LOSS_MAP[opt.functional](model, data[0], data[1])
        loss = LOSS_MAP[opt.functional](model, datI, datB)
        if opt.tau != 0:
            regularizer = 0
            for i , j in zip(model.parameters(), w0):
                regularizer += torch.sum(torch.pow((i - j),2))
            loss += opt.tau*regularizer                

        loss.backward()
        optimizer.step()
        loss_meter.add(loss.item())  # meters update
        scheduler.step()
        if opt.tau != 0:
            w0[:] = [i.data for i in model.parameters()]

        # if epoch % 100 == 0:
        #     log = 'Epoch: {:05d}, Loss: {:.5f}'
        #     print(log.format(epoch, torch.abs(torch.tensor(loss_meter.value()[0]))))
    
    # model.save(name = opt.model + opt.functional + f'Tau{opt.tau}Epoch{opt.max_epoch}.pt')
        
        if epoch % 100 == 0:
            test_err1 = eval(model, grid, exact1)
            test_err2 = eval(model, grid, exact2)
            if test_err1 < test_err2:
                if test_err1 < previous_err:
                    previous_err = test_err1
                    best_epoch = epoch
                    flag = True
                else:
                    pass
            else:
                if test_err2 < previous_err:
                    previous_err = test_err2
                    best_epoch = epoch
                    flag = False
                else:
                    pass
            log = 'Epoch: {:05d}, Loss: {:.5f}, Test: {:.5f}, Best Epoch: {:05d}, Which: {}'
            print(log.format(epoch, torch.abs(torch.tensor(loss_meter.value()[0])), previous_err.item(), best_epoch, flag))

        # if epoch % 100 == 0:
        #     test_err = eval(model, grid, sol)
        #     if test_err < previous_err:
        #         model.save(name = opt.model + f'Tau{opt.tau}Epoch{opt.max_epoch}.pt')


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



# just for testing, need to be modified
def make_plot(**kwargs):

    opt._parse(kwargs)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # configure model
    model = getattr(models, opt.model)().eval()

    model.load(osp.join(osp.dirname(osp.realpath(__file__)), 'checkpoints', 'ResNetallenTau10Epoch20000.pt'), dev = device)


    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'exact_sol', 'allen2dgrid.pt')
    #grid = torch.load(gridpath, map_location = device)
    grid = torch.load(path)
    
    with torch.no_grad():
        pred = model(grid)

    plot(pred)

    return None




def help():
    """
    Print out the help information： python file.py help
    """
    
    print("""
    usage : python file.py <function> [--args=value]
    <function> := train | make_plot | help
    Example: 
            python {0} train --lr=1e-5
            python {0} make_plot --load_model_path='...'
            python {0} help

    Avaiable args: please refer to config.py""".format(__file__))

    # from inspect import getsource
    # source = (getsource(opt.__class__))
    # print(source)

if __name__=='__main__':
    import fire
    fire.Fire()





