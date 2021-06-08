import torch
import torch.nn as nn
from torch import Tensor
import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
import models
from data import allencahn
from optim import Optim
from para_init import weight_init
from torchnet import meter
from config import opt
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable




def elliptical(x: Tensor, y: Tensor):
    # change this part according to your specific task
    m = nn.Tanh()
    return (-m(10*(torch.sqrt(x**2 + 4*y**2) - 0.5))).reshape((-1, 1))


def pretrain():
    # model configuration
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
    op = Optim(model.parameters(), opt)
    optimizer = op.optimizer

    # data initialization
    x = torch.linspace(-1, 1, 101)
    y = torch.linspace(-1, 1, 101)
    X, Y = torch.meshgrid(x, y)
    Z = torch.cat((X.flatten()[:, None], Y.flatten()[:, None]), dim=1)
    exact = elliptical(Z[:,0], Z[:,1])

    # train the initialization model
    for _ in range(7000 + 1):
        optimizer.zero_grad()
        datI = allencahn(num = 2500, boundary = False, device = device)
        datB = allencahn(num = 100, boundary = True, device = device)
        out_i = model(datI)
        out_b = model(datB)
        real_i = (elliptical(datI[:,0], datI[:,1])).reshape((-1,1))
        real_b = (elliptical(datB[:,0], datB[:,1])).reshape((-1,1))
        loss = torch.mean((out_i - real_i)**2)
        loss += torch.mean((out_b - real_b)**2)
        loss.backward()
        optimizer.step()

    # check the model and save it
    path = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'log', "")
    plt.figure()
    pred = model(Z)
    pred = pred.detach().numpy()
    pred = pred.reshape(101, 101)
    ax = plt.subplot(1, 1, 1)
    h = plt.imshow(pred, interpolation='nearest', cmap='rainbow',extent=[-1, 1, -1, 1],origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(h, cax=cax)
    plt.savefig(path + 'initialization.png')
    model.save('init.pt')







if __name__ == '__main__':
    pretrain()