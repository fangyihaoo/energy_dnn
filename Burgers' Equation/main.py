import torch
import torch.nn as nn                     # neural networks
import numpy as np
import time
from pyDOE import lhs         #Latin Hypercube Sampling
import scipy
from PINN import *

def main():
    
    'model setup'  
    torch.set_default_dtype(torch.float)
    torch.manual_seed(1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(device)

    if device == 'cuda': 
        print(torch.cuda.get_device_name())
        
    path = 'burgers_shock.mat'


    'Nu = 100 Boundary and initial points(random selection), Nf = 10000 Collocation points by lhs' 
    Nu, Nf = (100, 10000)

    X_f_train, X_u_train, u_train, X_u_test, u_true = DataPrep(path, Nu, Nf)

    'set bound'
    lb = X_u_test[0]           # [-1, 0]
    ub = X_u_test[-1]          # [1, 0.99]

    'define the model'
    model = Burgers(2, 20, 1, 6, lb, ub,X_f_train, X_u_train, u_train, X_u_test, u_true)
    model.to(device)

    'Optimizer LBFGS'
    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.1, max_iter = 500, max_eval = None, tolerance_grad = 1e-05, tolerance_change = 1e-09, history_size = 100, line_search_fn = 'strong_wolfe')

    start_time = time.time()
    optimizer.step(model.closure)
    elapsed = time.time() - start_time  

    print(f'Training time: {elapsed:.2f}')



    ' Model Accuracy '
    error_vec, u_pred = PINN.test()
    print(f'Test Error: {error_vec:.5f}')


    # 'Optimizer Adam'

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001,betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    # max_iter = 20000 # cause we are using full-batach so it's equivalent to epoach

    # for i in range(max_iter):
    #     loss = model.loss(X_f_train, X_u_train, u_train, X_u_test, u_true)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     if i % (max_iter//10) == 0:
    #         error_vec, _ = model.predict(X_u_test, u_true)
    #         print(f'In iteration {i}, the loss is {loss} and the error is {error_vec}')


if __name__ == "__main__":
    main()