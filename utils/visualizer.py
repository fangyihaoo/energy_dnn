import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import pi
import numpy as np
import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))



"""
Toy example
"""
# Relative L2 Norm plot

Evnn = torch.load('../log/toy/poiourmethod.pt', map_location=torch.device('cpu'))
Ritz = torch.load('../log/toy/poiDritz.pt', map_location=torch.device('cpu'))
pinn = torch.load('../log/toy/poipinn.pt', map_location=torch.device('cpu'))

cycEvnn = torch.load('../log/toy/poissoncycleourmethod.pt', map_location=torch.device('cpu'))
cycRitz = torch.load('../log/toy/poissoncycleDritz.pt', map_location=torch.device('cpu'))
cycPinn = torch.load('../log/toy/poissoncyclepinn.pt', map_location=torch.device('cpu'))
epoch = torch.arange(0, 50000)

# fig, ax = plt.subplots(1, 2, figsize=(15, 4))
# ax[0].spines['right'].set_visible(False)
# ax[0].spines['top'].set_visible(False)
# ax[1].spines['right'].set_visible(False)
# ax[1].spines['top'].set_visible(False)
# ax[0].set_title(r'$f = 1$')
# ax[1].set_title(r'$f = 2\sin{x}\cdot\cos{y}$')

# lines = []
# ax[0].set_yscale('log')
# lines = ax[0].plot(epoch[99:50000:100], cycEvnn[99:50000:100],  color= '#EE82EE' )
# lines += ax[0].plot(epoch[99:50000:100], cycRitz[99:50000:100], color='#5E5A80')
# lines += ax[0].plot(epoch[99:50000:100], cycPinn[99:50000:100], color='#69ECEB')
# ax[0].legend(lines[:3], ['EVNN', 'DeepRitz', 'PINN'], loc='upper right', frameon=False)
# ax[0].set_xlabel('epoch')
# ax[0].set_ylabel('Relative L2 Norm')

# lines = []
# ax[1].set_yscale('log')
# lines = ax[1].plot(epoch[99:50000:100], Evnn[99:50000:100],  color= '#EE82EE' )
# lines += ax[1].plot(epoch[99:50000:100], Ritz[99:50000:100], color='#5E5A80')
# lines += ax[1].plot(epoch[99:50000:100], pinn[99:50000:100], color='#69ECEB')
# ax[1].legend(lines[:3], ['EVNN', 'DeepRitz', 'PINN'], loc='upper right', frameon=False)
# ax[1].set_xlabel('epoch')
# ax[1].set_ylabel('Relative L2 Norm')


fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_title(r'$f = 1$')

lines = []
ax.set_yscale('log')
lines = ax.plot(epoch[99:50000:100], cycEvnn[99:50000:100],  color= '#EE82EE' )
lines += ax.plot(epoch[99:50000:100], cycRitz[99:50000:100], color='#5E5A80')
lines += ax.plot(epoch[99:50000:100], cycPinn[99:50000:100], color='#69ECEB')
ax.legend(lines[:3], ['EVNN', 'DeepRitz', 'PINN'], loc='upper right', frameon=False)
ax.set_xlabel('epoch')
ax.set_ylabel('Relative L2 Norm')

plt.savefig('../log/toy/l2cycle.png',pad_inches = 0.05, bbox_inches='tight')


fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_title(r'$f = 2\sin{x}\cdot\cos{y}$')

lines = []
ax.set_yscale('log')
lines = ax.plot(epoch[99:50000:100], Evnn[99:50000:100],  color= '#EE82EE' )
lines += ax.plot(epoch[99:50000:100], Ritz[99:50000:100], color='#5E5A80')
lines += ax.plot(epoch[99:50000:100], pinn[99:50000:100], color='#69ECEB')
ax.legend(lines[:3], ['EVNN', 'DeepRitz', 'PINN'], loc='upper right', frameon=False)
ax.set_xlabel('epoch')
ax.set_ylabel('Relative L2 Norm')

plt.savefig('../log/toy/l2sincos.png',pad_inches = 0.05, bbox_inches='tight')



"""
Heat equation
"""

# class OOMFormatter(ticker.ScalarFormatter):
#     def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
#         self.oom = order
#         self.fformat = fformat
#         ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
#     def _set_order_of_magnitude(self):
#         self.orderOfMagnitude = self.oom
#     def _set_format(self, vmin=None, vmax=None):
#         self.format = self.fformat
#         if self._useMathText:
#              self.format = r'$\mathdefault{%s}$' % self.format


# from config import opt
# import models
# ACTIVATION_MAP = {'relu' : nn.ReLU(),
#             'tanh' : nn.Tanh(),
#             'sigmoid': nn.Sigmoid(),
#             'leakyrelu': nn.LeakyReLU()}
# keys = {'FClayer':opt.FClayer, 
#         'num_blocks':opt.num_blocks,
#         'activation':ACTIVATION_MAP[opt.act],
#         'num_input':opt.num_input,
#         'num_output':opt.num_oupt, 
#         'num_node':opt.num_node}
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = getattr(models, opt.model)(**keys).eval()
# model.load(osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), 'log', 'heat', 'heat20.pt'), dev = device)

# pred = model(grid)
# pred = pred.detach().numpy()
# pred = pred.reshape(101, 101)
# pred = np.transpose(pred)
# plt.figure(figsize=(6,6))
# ax = plt.subplot(1, 1, 1)
# h = plt.imshow(pred, interpolation='nearest', cmap='viridis',extent=[0, 2, 0, 2],origin='lower', aspect='auto')
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# # plt.colorbar(h, cax=cax, format = ticker.FuncFormatter(fmt))
# plt.colorbar(h, cax=cax, format=OOMFormatter(-3, mathText=False))
# plt.savefig(osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), 'log', 'heat', 'heat100.png'), pad_inches = 0.1, bbox_inches='tight')


# grid = torch.load('../data/exact_sol/heatgrid.pt')
# exact = torch.load('../data/exact_sol/heatexact.pt')
# pred = model(grid)
# error = pred - exact[20]
# error = error.detach().numpy()
# error = error.reshape(101, 101)
# error = np.transpose(error)
# plt.figure(figsize=(6,6))
# ax = plt.subplot(1, 1, 1)
# h = plt.imshow(error, interpolation='nearest', cmap='RdBu',extent=[0, 2, 0, 2],origin='lower', aspect='auto')
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(h, cax=cax, format=OOMFormatter(-3, mathText=False))
# plt.savefig(osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), 'log', 'heat', 'heat20error.png'), pad_inches = 0.1, bbox_inches='tight')


# ======================================================================================
# ======================================================================================
# ======================================================================================
# ======================================================================================
# ======================================================================================
# ======================================================================================



# fig, axs = plt.subplots(2, 3)
# axs[0, 0].plot(x, y)
# axs[0, 0].set_title("t = 0.1")

# axs[0, 1].plot(x + 1, y + 1)
# axs[0, 1].set_title("t = 0.5")

# axs[0, 2].plot(x + 1, y + 1)
# axs[0, 2].set_title("t = 0.9")


# axs[1, 0].plot(x, y**2)
# axs[1, 1].plot(x + 2, y + 2)
# axs[1, 2].plot(x + 2, y + 2)


# fig.tight_layout()

# plt.figure(figsize=(10, 3.5))

# plt.subplot(1, 2, 1)
# plt.imshow(I, cmap='RdBu')
# plt.colorbar()

# plt.subplot(1, 2, 2)
# plt.imshow(I, cmap='RdBu')
# plt.colorbar(extend='both')
# plt.clim(-1, 1);










# if __name__ == '__main__':
    # from config import opt
    # import models
    # ACTIVATION_MAP = {'relu' : nn.ReLU(),
    #             'tanh' : nn.Tanh(),
    #             'sigmoid': nn.Sigmoid(),
    #             'leakyrelu': nn.LeakyReLU()}
    # keys = {'FClayer':opt.FClayer, 
    #         'num_blocks':opt.num_blocks,
    #         'activation':ACTIVATION_MAP[opt.act],
    #         'num_input':opt.num_input,
    #         'num_output':opt.num_oupt, 
    #         'num_node':opt.num_node}
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = getattr(models, opt.model)(**keys).eval()
    # model.load(osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), 'checkpoints', 'heat100.pt'), dev = device)    
    # x = torch.linspace(0, 2, 101)
    # y = torch.linspace(0, 2, 101)
    # X, Y = torch.meshgrid(x, y)
    # Z = torch.cat((X.flatten()[:, None], Y.flatten()[:, None]), dim=1)
    
    # pred = model(Z)
    # pred = pred.detach().numpy()
    # pred = pred.reshape(101, 101)
    # pred = np.transpose(pred)
    # plt.figure(figsize=(6,6))
    # ax = plt.subplot(1, 1, 1)
    # h = plt.imshow(pred, interpolation='nearest', cmap='rainbow',extent=[0, 2, 0, 2],origin='lower', aspect='auto')
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # plt.colorbar(h, cax=cax)
    # plt.savefig(osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))),'heat100.png'), pad_inches = 0.1, bbox_inches='tight')
    

    
    

    
    
    
    
    
    
    
    
    # error plot
# ----------------------------------------------------------------------
    # grid = torch.load('../data/exact_sol/heatgrid.pt')
    # exact = torch.load('../data/exact_sol/heatexact.pt')
    # pred = model(grid)
    # error = pred - exact[10]
    # error = error.detach().numpy()
    # error = error.reshape(101, 101)
    # error = np.transpose(error)
    # plt.figure(figsize=(6,6))
    # ax = plt.subplot(1, 1, 1)
    # h = plt.imshow(error, interpolation='nearest', cmap='rainbow',extent=[0, 2, 0, 2],origin='lower', aspect='auto')
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # plt.colorbar(h, cax=cax)
    # plt.savefig(osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))),'heat0.1error.png'), pad_inches = 0.1, bbox_inches='tight')
# ----------------------------------------------------------------------
    # allencahn300
    #allencahn2dloss1001
    # FClayer = 2

    # num_blocks = 3

    # num_input = 2

    # num_oupt = 1

    # num_node = 10
    
