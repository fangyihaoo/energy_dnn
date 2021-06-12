import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import pi
import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))



# def plot(pred):
#     '''
#     need to be modified to make it more general
#     '''
    
#     plt.figure()
#     if pred.is_cuda:
#         pred = pred.to('cpu').numpy()
#     else:
#         pred = pred.numpy()
#     pred = pred.reshape(201, 201)
#     ax = plt.subplot(1, 1, 1)
#     h = plt.imshow(pred, interpolation='nearest', cmap='rainbow',extent=[0, 1, 0, 1],origin='lower', aspect='auto')
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#     plt.colorbar(h, cax=cax)
#     plt.savefig('pred.png')

# def make_plot(**kwargs) -> None:
#     opt._parse(kwargs) 
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     # configure model
#     model = getattr(models, opt.model)().eval()
#     model.load(osp.join(osp.dirname(osp.realpath(__file__)), 'checkpoints', 'ResNetallenTau10Epoch20000.pt'), dev = device)
#     path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'exact_sol', 'allen2dgrid.pt')
#     #grid = torch.load(gridpath, map_location = device)
#     grid = torch.load(path)   
#     with torch.no_grad():
#         pred = model(grid)
#     plot(pred)
#     return None


if __name__ == '__main__':
    from config import opt
    import models
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = getattr(models, opt.model)(**keys).eval()
    model.load(osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), 'checkpoints', 'allencahn300.pt'), dev = device)
    x = torch.linspace(-1, 1, 101)
    y = torch.linspace(-1, 1, 101)
    X, Y = torch.meshgrid(x, y)
    Z = torch.cat((X.flatten()[:, None], Y.flatten()[:, None]), dim=1)
    pred = model(Z)
    pred = pred.detach().numpy()
    pred = pred.reshape(101, 101)
    plt.figure(figsize=(6,6))
    ax = plt.subplot(1, 1, 1)
    h = plt.imshow(pred, interpolation='nearest', cmap='rainbow',extent=[-1, 1, -1, 1],origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(h, cax=cax)
    plt.savefig(osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))),'allencahn300.png'))

    # allencahn300
    #allencahn2dloss1001
    # FClayer = 2

    # num_blocks = 3

    # num_input = 2

    # num_oupt = 1

    # num_node = 10