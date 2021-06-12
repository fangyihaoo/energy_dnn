import warnings

class DefaultConfig(object):
    '''
    default setting, can be changed via command line
    '''

    model = 'ResNet'

    functional = 'allenw'  # 'poi': PoiLoss   'allen' AllenCahnLoss

    FClayer = 2

    num_blocks = 3

    num_input = 2

    num_oupt = 1

    num_node = 10

    act = 'tanh'  # tanh,  relu,  sigmoid,  leakyrelu

    load_model_path = None

    pretrain = 'init.pt'

    exact = 'poiss2dexact.pt'

    grid = 'poiss2dgridpinn.pt'
    
    method = 'adam'

    max_epoch = 1001 # number of epoch

    lr = 1e-3 # initial learning rate

    lr_decay = 0.5 # lr = lr*lr_decay

    step_size = 1000

    momentum = 0.9 # Momentum for modified SGD

    nesterov = True # Nesterov momentum for SGD

    # alpha = 0.99 # alpha for RMSProp
    
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

