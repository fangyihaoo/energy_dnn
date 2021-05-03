import torch
import torch.nn as nn
import time


class BasicModule(nn.Module):
    """
    Encapsulation for 'save' and 'load' method
    """

    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name=str(type(self)) # default name

    def load(self, path):
        """
        load the model at specific path
        """
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        """
        save the model with the default name
        """
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pt')
        torch.save(self.state_dict(), name)