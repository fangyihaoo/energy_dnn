import math
import torch.optim as optim

class Optim(object):
    '''
    Make optimizer according to config.py
    '''

    def __init__(self, params, config):
        self.params = params  
        self.method = config.method
        self.lr = config.lr
        self.lr_decay = config.lr_decay
        self.weight_decay = config.weight_decay
        self.momentum = config.momentum
        self.nesterov = config.nesterov


    def _makeOptimizer(self):
        if self.method == 'sgd':
            if len(self.params) == 1:
                return optim.SGD(self.params, lr = self.lr, weight_decay = self.weight_decay, momentum = self.momentum, nesterov = self.nesterov)
            else:
                return optim.SGD(self.params, lr = self.lr,  momentum = self.momentum, nesterov = self.nesterov)

        elif self.method == 'adagrad':
            if len(self.params) == 1:
                return optim.Adagrad(self.params, lr = self.lr)
            else:
                return optim.Adagrad(self.params, lr = self.lr, weight_decay = self.weight_decay)

        elif self.method == 'rmsprop':
            if len(self.params) == 1:
                return optim.RMSProp(self.params, lr = self.lr, alpha = 0.9)
            else:
                return optim.RMSProp(self.params, lr = self.lr, alpha = 0.9, weight_decay = self.weight_decay)

        elif self.method == 'adam':
            if len(self.params) == 1:
                return optim.Adam(self.params, lr=self.lr)
            else:
                return optim.Adam(self.params, lr=self.lr, weight_decay = self.weight_decay)

        else:
            raise RuntimeError("Invalid optim method: " + self.method)
