import torch
import torch.nn as nn
from torch import Tensor
import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
import models
from data import allencahn, heat
from optim import Optim
from para_init import weight_init
from torchnet import meter
from config import opt
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.optim.lr_scheduler import StepLR
import numpy as np
from numpy import pi


def elliptical(x: Tensor, y: Tensor):
    # change this part according to your specific task
    m = nn.Tanh()
    return (-m(10*(torch.sqrt(x**2 + 4*y**2) - 0.5))).reshape((-1, 1))

def constant(x: Tensor, y: Tensor):
    N = x.shape[0]
    res = torch.tensor([-1.])
    return res.repeat((N,1))

def HeatInit(Z: Tensor, boundary: str = False):
    N = Z.shape[0]
    if boundary:
        return torch.tensor([0]).repeat(N).reshape((-1,1))
    else:
        data = torch.tensor([0]).repeat(N).reshape((-1,1))
        data[Z[:,1] <= 1] = 50
        return data
            
def HeatNew(Z: Tensor, boundary: str = False):
    N = Z.shape[0]
    if boundary:
        return torch.tensor([0]).repeat(N).reshape((-1,1))
    else:
        data = (torch.sin(0.5*pi*Z[:,0])*torch.sin(0.5*pi*Z[:,1])).unsqueeze_(1)
        return data
    


def pretrain():
    # heat model configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    model.to(device)
    model.apply(weight_init)
    op = Optim(model.parameters(), opt)
    optimizer = op.optimizer
    scheduler = StepLR(optimizer, step_size=opt.step_size, gamma=opt.lr_decay)

    # data initialization
    # x = torch.linspace(0, 2, 101)
    # y = torch.linspace(0, 2, 101)
    # X, Y = torch.meshgrid(x, y)
    # Z = torch.cat((X.flatten()[:, None], Y.flatten()[:, None]), dim=1)
    # exact = constant(Z[:,0], Z[:,1])
    
    
    path = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'data', 'exact_sol' ,"")
    grid = torch.load(path + 'heatgrid.pt', map_location = device)
    exact = torch.load(path + 'heatexact.pt', map_location = device)
    
    # train the initialization model
    for _ in range(10000 + 1):
        optimizer.zero_grad()
        datI = heat(num = 5000, boundary = False, device = device)
        datB = heat(num = 1000, boundary = True, device = device)
        out_i = model(datI)
        out_b = model(datB)
        real_i = HeatNew(datI, boundary = False)
        real_b = HeatNew(datB, boundary = True)
        loss = torch.mean((out_i - real_i)**2)
        loss += 50*torch.mean((out_b - real_b)**2)
        if _ % 500 == 0:
            print(loss)
        loss.backward()
        optimizer.step()
        scheduler.step()

    # check the model and save it
    path = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'log', "")
    plt.figure()
    pred = model(grid)
    pred = pred.detach().numpy()
    pred = pred.reshape(101, 101)
    pred = np.transpose(pred)
    ax = plt.subplot(1, 1, 1)
    h = plt.imshow(pred, interpolation='nearest', cmap='rainbow',extent=[0, 2, 0, 2],origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(h, cax=cax)
    plt.savefig(path + 'heat.png')
    model.save('heat.pt')







if __name__ == '__main__':
    pretrain()