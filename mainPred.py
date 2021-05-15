from config import opt
import torch 
import models
from data import Poisson
from utils import Optim
from utils import criterion
from utils import weight_init
from utils import plot
from utils import seed_setup
from torch.utils.data import DataLoader
from torchnet import meter
from tqdm import tqdm


def train(**kwargs):

    opt._parse(kwargs)

    # setting device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load exact solution
    gridpath = './data/exact_sol/poiss2dgrid.pt'
    solpath = './data/exact_sol/poiss2dexact.pt'
    grid = torch.load(gridpath, map_location = device)
    sol = torch.load(solpath, map_location = device)

    #  meters
    test_meter = meter.AverageValueMeter()
    epoch_meter = meter.AverageValueMeter()

    for i in range(opt.ite + 1):

        # configure model
        model = getattr(models, opt.model)()
        if opt.load_model_path:
            model.load(opt.load_model_path)
        model.to(device)

        model.apply(weight_init)

        # optimizer
        op = Optim(model.parameters(), opt)
        optimizer = op._makeOptimizer()

        # regularization
        if opt.tau:
            w0 = [torch.zeros_like(p.data) for p in model.parameters()]   # the initial w0 value

        # previous_loss = torch.tensor(10000)
        previous_err = torch.tensor(10000)
        best_epoch = 0

        # train
        for epoch in range(opt.max_epoch + 1):
            
            datI = Poisson(num = 100, boundary = False, device = device)
            datB = Poisson(num = 25, boundary = True, device = device)

            datI_loader = DataLoader(datI, 100, shuffle=True) # make sure that the dataloders are the same len for datI and datB
            datB_loader = DataLoader(datB, 10, shuffle=True)

            for data in zip(datI_loader, datB_loader):

                # train model 
                optimizer.zero_grad()
                loss = criterion(model, data[0], data[1])

                if opt.tau:
                    regularizer = torch.tensor(0.0)
                    for i , j in zip(model.parameters(), w0):
                        regularizer = regularizer + torch.sum(torch.pow((i - j),2)) # not sure whether inplace addition appropriate here
                    loss = loss + opt.tau*regularizer                

                loss.backward()
                optimizer.step()

            # update w0
            if opt.tau:
                w0[:] = [i.data for i in model.parameters()]
        
            if epoch % 100 == 0:
                test_err = val(model, grid, sol)
                if test_err < previous_err:
                    previous_err = test_err
                    best_epoch = epoch
        test_meter.add(previous_err)
        epoch_meter.add(best_epoch)

    print(f'Mean MSE: {test_meter.value()[0]:.5f},  Std: {test_meter.value()[1]:.5f},  Mean Epoach: {epoch_meter.value()[0]: 05d}')


        
        # update learning rate
        # if loss_meter.value()[0] > previous_loss:          
        #     lr = lr * opt.lr_decay
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr



# validation function
@torch.no_grad()
def val(model, data, sol):
    """
    validation part
    """
    model.eval()
    
    pred = torch.flatten(model(data))

    err  = torch.mean(torch.pow(pred - sol, 2))

    model.train()

    return err





# just for testing, need to be modified
def make_plot(**kwargs):

    opt._parse(kwargs)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # configure model
    model = getattr(models, opt.model)().eval()
    if opt.load_model_path:
        model.load(opt.load_model_path, dev = device)


    gridpath = './data/exact_sol/poiss2dgrid.pt'
    #grid = torch.load(gridpath, map_location = device)
    grid = torch.load(gridpath)
    
    with torch.no_grad():
        pred = model(grid)

    plot(pred)

    return None




def help():
    """
    Print out the help information： python file.py help
    """
    
    print("""
    usage : python file.py <function> [--args=value]
    <function> := train | make_plot | help
    example: 
            python {0} train --weight_decay='1e-5' --lr=0.01
            python {0} make_plot --load_model_path='...'
            python {0} help
    avaiable args:""".format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)

if __name__=='__main__':
    import fire
    fire.Fire()
