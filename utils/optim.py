import torch.optim as optim

class Optim(object):
    '''
    Make optimizer according to config.py
    '''

    def __init__(self, params, config):
        self.params = params  
        self.method = config.method
        self.lr = config.lr
        # self.lr_decay = config.lr_decay
        # self.momentum = config.momentum
        # self.nesterov = config.nesterov
        self._makeOptimizer()

    def _makeOptimizer(self):
        if self.method == 'adagrad':
            self.optimizer =  optim.Adagrad(self.params, lr = self.lr)

        elif self.method == 'rmsprop':
            self.optimizer =  optim.RMSProp(self.params, lr = self.lr)

        elif self.method == 'adam':
            self.optimizer =  optim.Adam(self.params, lr=self.lr)
        
        elif self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr = self.lr)

        else:
            raise RuntimeError("Invalid optim method: " + self.method)

