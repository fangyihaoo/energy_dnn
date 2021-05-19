import warnings

class DefaultConfig(object):
    '''
    default setting, can be changed via command line
    '''

    model = 'ResNet'

    loss = 'DRM'  # 'DGM'

    load_model_path = None
    #'checkpoints/model.pth' # path for trained model
    
    method = 'adam'

    max_epoch = 50000 # number of epoch

    lr = 0.0001 # initial learning rate

    lr_decay = 0.95 # when val_loss increase, lr = lr*lr_decay

    # weight_decay = 0 # initial peanality for the L2

    momentum = 0.9 # Momentum for modified SGD

    nesterov = True # Nesterov momentum for SGD

    alpha = 0.99 # alpha for RMSProp
    
    tau = None # 1/tau

    ite = 2 # number of iteration


    def _parse(self, kwargs):
        '''
        update parameters according to user preference
        '''
        for k,v in kwargs.items():
            if not hasattr(self,k):
                warnings.warn("Warning: opt has not attribut %s" %k)
            setattr(self,k,v)

        print('user config:')
        for k,v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                if '_parse' in k:
                    continue
                else:
                    print(k,getattr(self,k))


opt = DefaultConfig()

