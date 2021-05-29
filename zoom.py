from config import opt
import torch 
from torch.optim.lr_scheduler import StepLR
import models
from data import Poisson
from utils import Optim
from utils import PoiLoss
from utils import weight_init
from utils import plot
from utils import seed_setup
from torch.utils.data import DataLoader
from torchnet import meter


def train(**kwargs):

    opt._parse(kwargs)

    # setting device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load exact solution
    gridpath = './data/exact_sol/poiss2dgrid.pt'
    solpath = './data/exact_sol/poiss2dexact.pt'
    grid = torch.load(gridpath, map_location = device)
    sol = torch.load(solpath, map_location = device)



    # seed_setup() # fix seed


    # configure model
    model = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model.to(device)

    model.apply(weight_init)


    # optimizer
    op = Optim(model.parameters(), opt)
    optimizer = op._makeOptimizer()
    scheduler = StepLR(optimizer, step_size=opt.step_size, gamma=opt.lr_decay)

    #  meters
    loss_meter = meter.AverageValueMeter()

    # regularization
    if opt.tau:
        w0 = [torch.zeros_like(p.data) for p in model.parameters()]   # the initial w0 value


    # previous_loss = torch.tensor(10000)
    previous_err = torch.tensor(10000)
    best_epoch = 0

    # train
    for epoch in range(opt.max_epoch + 1):
        
        loss_meter.reset()

        datI = Poisson(num = 100, boundary = False, device = device)
        datB = Poisson(num = 25, boundary = True, device = device)

        datI_loader = DataLoader(datI, 10, shuffle=True) # make sure that the dataloders are the same len for datI and datB
        datB_loader = DataLoader(datB, 10, shuffle=True)

        for data in zip(datI_loader, datB_loader):

            # train model 
            optimizer.zero_grad()
            loss = PoiLoss(model, data[0], data[1])

            if opt.tau:
                regularizer = torch.tensor(0.0)
                for i , j in zip(model.parameters(), w0):
                    regularizer = regularizer + torch.sum(torch.pow((i - j),2)) # not sure whether inplace addition appropriate here
                loss = loss + opt.tau*regularizer                

            loss.backward()
            loss_meter.add(loss.item())  # meters update
            optimizer.step()
            
        scheduler.step()
        # update w0
        if opt.tau:
            w0[:] = [i.data for i in model.parameters()]
        

        if epoch % 100 == 0:
            test_err = val(model, grid, sol)
            if test_err < previous_err:
                previous_err = test_err
                best_epoch = epoch
            log = 'Epoch: {:05d}, Loss: {:.5f}, Test: {:.5f}, Best Epoch: {:05d}'
            print(log.format(epoch, torch.abs(torch.tensor(loss_meter.value()[0])), test_err.item(), best_epoch))


# validation function

@torch.no_grad()
def val(model, data, sol):
    """
    validation part
    """
    model.eval()
    pred = model(data)
    err  = torch.pow(torch.mean(torch.pow(pred - sol, 2))/torch.mean(torch.pow(sol, 2)), 0.5)
    model.train()

    return err




if __name__=='__main__':
    import fire
    fire.Fire()
