from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import pi




def plot(pred):
    '''
    need to be modified to make it more general
    '''
    
    plt.figure()
    if pred.is_cuda:
        pred = pred.to('cpu').numpy()
    else:
        pred = pred.numpy()
    pred = pred.reshape(1001, 1001)
    ax = plt.subplot(1, 1, 1)
    h = plt.imshow(pred, interpolation='nearest', cmap='rainbow',
                   extent=[0, pi, -2./pi, 2./pi],
                   origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(h, cax=cax)
    plt.savefig('pred.png')

