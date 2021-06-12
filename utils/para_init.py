import torch
import torch.nn as nn
import torch.nn.init as init


@torch.no_grad()
def weight_init(m):
    r"""Implement different weight initilization method (bias are set to zero)

    Args:
        m: model layer object
            please refer to 'https://pytorch.org/docs/stable/nn.init.html' for details

    Example:
        >>> model.apply(weight_init)
    """

    if isinstance(m, nn.Linear):
        gain = init.calculate_gain('tanh') # change this part according to your activation function
        init.xavier_uniform_(m.weight.data, gain=gain)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    
    # if isinstance(m, nn.Linear):
    #     gain = init.calculate_gain('relu') 
    #     init.xavier_normal_(m.weight.data, gain=gain)
    #     if m.bias is not None:
    #         torch.nn.init.zeros_(m.bias)

    # if isinstance(m, nn.Linear):
    #     init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
    #     if m.bias is not None:
    #         init.zeros_(m.bias)

    # if isinstance(m, nn.Linear):
    #     init.uniform_(m.weight.data, a=0.0, b=1.0) # change a, b according to your setting
    #     if m.bias is not None:
    #         init.zeros_(m.bias)

    # if isinstance(m, nn.Linear):
    #     init.normal_(m.weight.data, mean=0.0, std=1.0) # change mean, std according to your settin
    #     if m.bias is not None:
    #         init.zeros_(m.bias)


















if __name__ == '__main__':
    pass