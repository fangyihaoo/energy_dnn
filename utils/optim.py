import torch.optim as optim
from torch.optim import Optimizer
import torch
from torch import Tensor
from typing import List


class Optim(object):
    r"""Initilize different optimizer

    Args:
        params: model parameters
        opt: self-defined config file.  see config.py file in main folder

    Example:
        >>> op = Optim(model.parameters(), opt)
        >>> optimizer = op.optimizer
    """

    def __init__(self, params, config):
        self.params = params  
        self.method = config.method
        self.lr = config.lr
        self.momentum = config.momentum
        self.nesterov = config.nesterov
        self._makeOptimizer()

    def _makeOptimizer(self):
        if self.method == 'adagrad':
            self.optimizer =  optim.Adagrad(self.params, lr = self.lr)

        elif self.method == 'rmsprop':
            self.optimizer =  optim.RMSProp(self.params, lr = self.lr)

        elif self.method == 'adam':
            self.optimizer =  optim.Adam(self.params, lr=self.lr, weight_decay = 1e-5)
        
        elif self.method == 'adadelta':
            self.optimizer =  optim.Adadelta(self.params)

        elif self.method == 'BB':
            self.optimizer = BB(self.params, lr = self.lr)
        
        elif self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr = self.lr, momentum=self.momentum, nesterov=self.nesterov)

        else:
            raise RuntimeError("Invalid optim method: " + self.method)




class BB(Optimizer):
    r"""Implements Modified Barzilai-Borwein Algorithm
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-5)
        alpha (float, optional): normalization rate of the gradient

    """

    def __init__(self, params, lr = 1e-8, alpha = 1):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= alpha:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr = lr, alpha = alpha)
        super(BB, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(BB, self).__setstate__(state)

    @torch.no_grad()
    def step(self, jump: int = 11, closure = None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            jump (int， optional): warm up stage. 
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            oldparams = []
            oldgrads = []
            lr = group['lr']
            alpha = group['alpha']
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    grads.append(p.grad)
                    state = self.state[p]
                    # lazy initialization
                    if len(state) == 0:
                        state['oldparam'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['oldgrad'] = torch.zeros_like(p.grad, memory_format=torch.preserve_format)
                    oldparams.append(state['oldparam'])
                    oldgrads.append(state['oldgrad'])
 
            bb(params_with_grad,
                   grads,
                   oldparams,
                   oldgrads,
                   lr=lr,
                   alpha = alpha,
                   jump=jump)

        return loss

def bb(params: List[Tensor],
        grads: List[Tensor],
        oldparams: List[Tensor],
        oldgrads: List[Tensor],
        *,
        lr: float,
        alpha: float,
        jump: int) -> None:
    r"""
    Functional API to perform BB updating algorithm 
    """
    lr = lr
    #update the learning rate if jump greater than 20
    if jump >= 10: 
        step_I = 0.
        sk_sum = 0
        skyk_sum = 0
        device = params[0].grad.device
        gradnorm = torch.norm(torch.stack([torch.norm(p.grad.detach()).to(device) for p in params]))
        for i, param in enumerate(params):
            yk = grads[i] - oldgrads[i]
            sk = param.detach() - oldparams[i]
            sk_sum += torch.sum(sk*sk)
            skyk_sum += torch.sum(yk*sk)
        step_I = sk_sum/(skyk_sum + 1e-8)
        lr = min(step_I, alpha/gradnorm)
    # update the parameters
    for i, param in enumerate(params):
        grad = grads[i]
        oldparam = oldparams[i]
        oldgrad = oldgrads[i]
        oldparam.copy_(param.detach())
        oldgrad.copy_(grad)
        param.add_(grad, alpha = -lr)
