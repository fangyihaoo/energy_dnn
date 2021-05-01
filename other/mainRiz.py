import torch
from torch import Tensor
import torch.nn as nn
from pyDOE import lhs
from torch.utils.data import Dataset, DataLoader
from typing import Type, Any, Callable, Union, List, Optional
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from DeepRitz import *



'model setup'  
torch.set_default_dtype(torch.float)
torch.manual_seed(1234)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

if device == 'cuda': 
    print(torch.cuda.get_device_name())


'mini-batach version'
def main():
    
    model = ResNet(ResBlock,num_blocks = 4)
    
    model = model.to(device)
    
    epochs = 50000
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001,betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    
    best_loss, best_epoch = 1000, 0
    
    for epoch in range(epochs+1):
        
        dat_i, dat_b = GenDat(device)

        datset = Poisson(dat_i, dat_b)

        loader = DataLoader(datset, batch_size=40, shuffle=True)
        
        for data in loader:
            
            los = Loss(model, data)
            
            optimizer.zero_grad()
            
            los.backward()
            
            optimizer.step()
        
        if epoch % 100 == 0:
            print('epoch:', epoch, 'los:', los.item())
            if epoch > int(4 * epochs / 5):
                if torch.abs(los) < best_loss:
                    best_loss = torch.abs(los).item()
                    best_epoch = epoch
                    torch.save(model.state_dict(), 'new_best_deep_ritz1.mdl')
    print('best epoch:', best_epoch, 'best loss:', best_loss)
    
    model.load_state_dict(torch.load('new_best_deep_ritz1.mdl'))
    print('load from ckpt!')
    
    with torch.no_grad():
        x1 = torch.linspace(-1, 1, 1001)
        x2 = torch.linspace(-1, 1, 1001)
        X, Y = torch.meshgrid(x1, x2)
        Z = torch.cat((Y.flatten()[:, None], Y.T.flatten()[:, None]), dim=1)
        # if 2 < m:
        #     y = torch.zeros(Z.shape[0], m - 2)
        #     Z = torch.cat((Z, y), dim=1)
        Z = Z.to(device)
        
        pred = model(Z)
 
    
    plt.figure()
    pred = pred.cpu().numpy()
    pred = pred.reshape(1001, 1001)
    ax = plt.subplot(1, 1, 1)
    h = plt.imshow(pred, interpolation='nearest', cmap='rainbow',
                   extent=[-1, 1, -1, 1],
                   origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(h, cax=cax)
    plt.savefig('test.png')

    
    

'full-batch version'
# def main():
    
#     model = ResNet(ResBlock,num_blocks = 4)
    
#     model = model.to(device)
    
#     epochs = 50000
    
#     optimizer = torch.optim.Adam(model.parameters(), lr=3e-3,betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    
#     best_loss, best_epoch = 1000, 0
    
#     for epoch in range(epochs+1):
        
#         dat_i, dat_b = GenDat(device)
        
#         los = Loss(model, dat_i, dat_b)

#         optimizer.zero_grad()

#         los.backward()

#         optimizer.step()
        
#         if epoch % 100 == 0:
#             print('epoch:', epoch, 'los:', los.item())
#             if epoch > int(4 * epochs / 5):
#                 if torch.abs(los) < best_loss:
#                     best_loss = torch.abs(los).item()
#                     best_epoch = epoch
#                     torch.save(model.state_dict(), 'new_best_deep_ritz1.mdl')
#     print('best epoch:', best_epoch, 'best loss:', best_loss)
    
#     model.load_state_dict(torch.load('new_best_deep_ritz1.mdl'))
#     print('load from ckpt!')
    
#     with torch.no_grad():
#         x1 = torch.linspace(-1, 1, 1001)
#         x2 = torch.linspace(-1, 1, 1001)
#         X, Y = torch.meshgrid(x1, x2)
#         Z = torch.cat((Y.flatten()[:, None], Y.T.flatten()[:, None]), dim=1)
#         # if 2 < m:
#         #     y = torch.zeros(Z.shape[0], m - 2)
#         #     Z = torch.cat((Z, y), dim=1)
#         Z = Z.to(device)
        
#         pred = model(Z)
 
    
#     plt.figure()
#     pred = pred.cpu().numpy()
#     pred = pred.reshape(1001, 1001)
#     ax = plt.subplot(1, 1, 1)
#     h = plt.imshow(pred, interpolation='nearest', cmap='rainbow',
#                    extent=[-1, 1, -1, 1],
#                    origin='lower', aspect='auto')
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#     plt.colorbar(h, cax=cax)
#     plt.savefig('test.png') 
    
    
    

if __name__ == '__main__':
    main()