from config import opt
import torch 
import models
from data import Poisson
from utils import Optim
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
    best_epoch = 0

    # train
    for epoch in tqdm(range(opt.max_epoch + 1)):
        
        loss_meter.reset()

        datI = Poisson(device = device)
        datB = Poisson(num = 25, boundary = True, device = device)

        datI_loader = DataLoader(datI, 100, shuffle=True) # make sure that the dataloders are the same len for datI and datB
        datB_loader = DataLoader(datB, 10, shuffle=True)

        for i, data in enumerate(zip(datI_loader, datB_loader)):

            # train model 
            optimizer.zero_grad()
            loss = criterion(model, data[0], data[1])
            loss.backward()
            optimizer.step()
            
            
            # meters update and visualize
            loss_meter.add(loss.item())
        
        # prediction error
        pred = model(grid)
        test_err = torch.mean((pred - sol)**2)

        if epoch % 100 == 0:
            if epoch > int(4 * epochs / 5):
                if torch.abs(loss_meter.value()[0]) < previous_loss:
                    previous_loss = torch.abs(loss_meter.value()[0])
                    best_epoch = epoch
                    model.save()        
        
        # update learning rate
        # if loss_meter.value()[0] > previous_loss:          
        #     lr = lr * opt.lr_decay
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr
        
        log = 'Epoch: {:05d}, Loss: {:.6f}, Test: {:.6f}
        print(log.format(epoch, loss.item(), test_err))

        previous_loss = loss_meter.value()[0]


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
    <function> := train | test | help
    example: 
            python {0} train --weight_decay='env0701' --lr=0.01
            python {0} help
    avaiable args:""".format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)

if __name__=='__main__':
    import fire
    fire.Fire()
