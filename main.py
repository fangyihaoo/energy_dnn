from config import opt
import torch 
import models
from data import Poisson
from utils import Optim
from utils import criterion
from utils import weight_init
from utils import plot
from torch.utils.data import DataLoader
from torchnet import meter
from tqdm import tqdm



# # test function, no need for this model
# @torch.no_grad() 
# def test(**kwargs):
#     pass



# # export the ouput as csv file, no need for this model
# def write_csv(results,file_name):
#     import csv
#     with open(file_name,'w') as f:
#         writer = csv.writer(f)
#         writer.writerow(['test_err'])
#         writer.writerows(results)


def train(**kwargs):
    opt._parse(kwargs)

    # setting device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load exact solution
    gridpath = './data/exact_sol/poiss2dgrid.pt'
    solpath = './data/exact_sol/poiss2dexact.pt'
    grid = torch.load(gridpath, map_location = device)
    sol = torch.load(solpath, map_location = device)


    # configure model
    model = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model.to(device)

    model.apply(weight_init)


    w0 = [torch.zeros_like(p.data) for p in model.parameters()]   # the initial value
    
    # optimizer
    # we only apply L2 penalty on weight
    # weight_p, bias_p = [], []
    # for name, p in model.named_parameters():
    #     if 'bias' in name:
    #         bias_p += [p]
    #     else:
    #         weight_p += [p]

    # op = Optim([
    #     {'params': weight_p, 'weight_decay': opt.weight_decay},
    #     {'params': bias_p, 'weight_decay':0}
    #     ], opt)

    # optimizer
    # penalize all of the parameters
    op = Optim(model.parameters(), opt)
    optimizer = op._makeOptimizer()

    #  meters
    loss_meter = meter.AverageValueMeter()
    previous_loss = 1e10
    # previous_err = 1e10
    best_epoch = 0

    # train
    for epoch in range(opt.max_epoch + 1):
        
        loss_meter.reset()

        datI = Poisson(device = device)
        datB = Poisson(num = 25, boundary = True, device = device)

        datI_loader = DataLoader(datI, 100, shuffle=True) # make sure that the dataloders are the same len for datI and datB
        datB_loader = DataLoader(datB, 10, shuffle=True)

        for data in zip(datI_loader, datB_loader):

            # train model 
            optimizer.zero_grad()
            loss = criterion(model, data[0], data[1])

            regularizer = torch.tensor(0.0)
            for i , j in zip(model.parameters(), w0):
                regularizer = regularizer + torch.sum(torch.pow((i - j),2)) # not sure whether inplace addition appropriate here
            
            loss = loss + opt.tau*regularizer

            loss.backward()
            optimizer.step()
            
            loss_meter.add(loss.item())  # meters update
        
        w0[:] = [i.data for i in model.parameters()]
        
        # prediction error
        with torch.no_grad():
            pred = torch.flatten(model(grid))
            test_err = torch.mean(torch.pow((pred - sol),2))

        # save model with least abs loss
        if epoch % 100 == 0:
            if epoch > int(4 * opt.max_epoch / 5):
                if torch.abs(loss_meter.value()[0]) < best_loss:
                    best_loss = torch.abs(loss_meter.value()[0])
                    best_epoch = epoch
                    model.save(name = 'new_best_' + f'Tau{opt.tau}' + '.pt')

        # save model with least abs error
        # if epoch % 100 == 0:
        #     if epoch > int(4 * opt.max_epoch / 5):
        #         if test_err < previous_err:
        #             previous_err = test_err
        #             best_epoch = epoch
        #             model.save(name = 'checkpoints/best.pt')

        
        # update learning rate
        # if loss_meter.value()[0] > previous_loss:          
        #     lr = lr * opt.lr_decay
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr

        if epoch % 100 == 0:
            log = 'Epoch: {:05d}, Loss: {:.5f}, Test: {:.5f}, Best Epoch: {:05d}'
            print(log.format(epoch, loss_meter.value()[0], test_err, best_epoch))


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

# @torch.no_grad()
# def val(model,dataloader):
#     pass
#     """
#     no need for this model
#     """


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
