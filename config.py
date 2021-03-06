import warnings

class DefaultConfig(object):
    '''
    default setting, can be changed via command line
    '''

    model = 'ResNet'

    functional = 'poi'  # 'poi': PoiLoss   'allen' AllenCahnLoss

    FClayer = 2

    num_blocks = 2

    num_input = 2

    num_oupt = 1

    num_node = 10

    act = 'tanh'  # tanh,  relu,  sigmoid,  leakyrelu

    load_model_path = None

    pretrain = None

    exact = 'poiss2dexact.pt'

    grid = 'poiss2dgrid.pt'

    method = 'adam'

    max_epoch = 100 # number of epoch

    lr = 1e-2 # initial learning rate

    lamda = 1e-4

    lam = 1

    lr_decay = 0.5 # lr = lr*lr_decay

    step_size = 2000

    dimension = 10

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

