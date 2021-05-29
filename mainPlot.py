import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR 
import models
from data import Poisson
from utils import Optim
from utils import PoiLoss
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
    exactpath = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'exact_sol', opt.exact)
    gridpath = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'exact_sol', opt.grid)
    grid = torch.load(gridpath, map_location = device)
    exact = torch.load(exactpath, map_location = device)

    # configure model
    FUNCTION_MAP = {'relu' : nn.ReLU(),
                    'tanh' : nn.Tanh(),
                    'sigmoid': nn.Sigmoid(),
                    'leakyrelu': nn.LeakyReLU()}
    keys = {'FClayer':opt.FClayer, 
            'num_blocks':opt.num_blocks,
            'activation':FUNCTION_MAP[opt.act],
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
        datI = Poisson(num = 1000, boundary = False, device = device)
        datB = Poisson(num = 100, boundary = True, device = device)
        datI_loader = DataLoader(datI, 100, shuffle=True) # make sure that the dataloders are the same len for datI and datB
        datB_loader = DataLoader(datB, 10, shuffle=True)

        for data in zip(datI_loader, datB_loader):
            optimizer.zero_grad()
            loss = PoiLoss(model, data[0], data[1])
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
        
        if epoch % 100 == 0:
            test_err = eval(model, grid, exact)
            if test_err < previous_err:
                previous_err = test_err
                best_epoch = epoch
            log = 'Epoch: {:05d}, Loss: {:.5f}, Test: {:.5f}, Best Epoch: {:05d}'
            print(log.format(epoch, torch.abs(torch.tensor(loss_meter.value()[0])), test_err.item(), best_epoch))

        # if epoch % 100 == 0:
        #     test_err = eval(model, grid, sol)
        #     if test_err < previous_err:
        #         model.save(name = opt.model + f'Tau{opt.ma}Epoch{opt.max_epoch}.pt')


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
    if opt.load_model_path:
        model.load(opt.load_model_path, dev = device)


    gridpath = './data/exact_sol/poiss2dgrid.pt'
    #grid = torch.load(gridpath, map_location = device)
    grid = torch.load(gridpath)
    
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










    # optimizer
    # we only apply L2 penalty on weight
    # weight_p, bias_p = [], []
    # for name, p in model.named_parameters():
    #     if 'bias' in name:
    #         bias_p += [p]
    #     else:
    #         weight_p += [p]

    # op = Optim([
    #     {'params': weight_p, 'weight_decay': opt.weight_decay},
    #     {'params': bias_p, 'weight_decay':0}
    #     ], opt)
