import torch
import torch.nn as nn                     # neural networks
import numpy as np
from pyDOE import lhs         #Latin Hypercube Sampling
import scipy

'Network structure'
class PINN(nn.Module):
    def __init__(self, num_input, num_neuron, num_output, num_layer, lowerboud, upperbound):
        super(PINN,self).__init__()
        
        'Initilize iter'
        self.iter = 0
        
        'Domain bound'
        self.lb = lowerboud
        self.ub = upperbound
        
        'Input layer' 
        self.input = nn.Linear(num_input,num_neuron)
        
        'Fully connected blocks'     
        self.linears_list = [nn.Linear(num_neuron, num_neuron) for i in range(num_layer)]
        self.acti_list = [nn.Tanh() for i in range(num_layer)]
        self.block = nn.Sequential(*[item for pair in zip(self.linears_list, self.acti_list + [0])for item in pair])
        
        'Output layer'        
        self.output = nn.Linear(num_neuron,num_output)
        
        'Xavier Normal Initialization'        
        nn.init.xavier_normal_(self.input.weight.data, gain=1.0)
        for m in self.block:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data,  gain=1.0)
        nn.init.xavier_normal_(self.output.weight.data, gain=1.0)
        
    def forward(self, x):
        x = (x - self.lb)/(self.ub - self.lb)
        x = self.input(x)
        x = self.block(x)
        x = self.output(x)
        return x
    
'Burgers equation'
class Burgers(PINN):
    def __init__(self,num_input, num_neuron, num_output, num_layer, lowerboud, upperbound, X_f_train, X_u_train, u_train, X_u_test, u_true)
        super(PINN, self).__init__(num_input, num_neuron, num_output, num_layer, lowerboud, upperbound)
        
        'Set up the data'
        self.X_f_train = X_f_train
        self.X_u_train = X_u_train
        self.u_train   = u_train
        self.X_u_test  = X_u_test
        self.u_true    = u_true
    
    'Loss function with two parts'  
    def loss(self, X_f_train, X_u_train, u_train, X_u_test, u_true):
        'Boundary loss and initial loss, UB'
        UB = self.forward(X_u_train)
        lossU   = ((UB - u_train)**2).mean()            

        'Collocation loss (UF), with strong form constraint' 
        g = X_f_train.clone()
        g.requires_grad = True
        UF = self.forward(g) 
        tmp   = torch.autograd.grad(outputs = UF, inputs = g, grad_outputs = torch.ones(UF.size()), retain_graph=True, create_graph=True)[0]
        UFx   = tmp[:,[0]]
        UFt   = tmp[:,[1]]
        UFxx  = torch.autograd.grad(outputs = tmp, inputs = g, grad_outputs = torch.ones(tmp.size()), create_graph=True)[0][:,[0]]
        U     = self.forward(X_f_train)
        F     = UFt + U*UFx - (0.01/np.pi)*UFxx
        lossF = (F**2).mean()

        'Combing two parts of error'
        loss  = lossU + lossF

        return loss
    
    'Prediction function'
    def predict(self, X_u_test, u_true):
        
        u_pred = self.forward(X_u_test)

        error_vec = torch.linalg.norm((u_true-u_pred),2)/torch.linalg.norm(u_true,2)        # Relative L2 Norm of the error (Vector)

        u_pred = u_pred.cpu().detach().numpy()

        u_pred = np.reshape(u_pred,(256,100),order='F')

        return error_vec, u_pred
    
    'API for the optimizer'
    def closure(self):
        
        optimizer.zero_grad()
        loss = self.loss(X_f_train, X_u_train, u_train, X_u_test, u_true)
        loss.backward()
                
        self.iter += 1
        
        if self.iter % 100 == 0:

            error_vec, _ = self.predict( X_u_test, u_true)
            
            print(f'In {self.iter} iteration,the Loss is {loss} and the Error is {error_vec}')

        return loss
        


'Data processing'

def DataPrep(path, N_u, N_f):
    'load the data'
    data = scipy.io.loadmat(path) 
    
    'extract and reshape the data'
    x = data['x']                                   # 256 points between -1 and 1 [256x1]
    t = data['t']                                   # 100 time points between 0 and 1 [100x1] 
    usol = data['usol']                             # solution of 256x100 grid points
    X, T = np.meshgrid(x,t)                         # makes 2 arrays X and T such that u(X[i],T[j])=usol[i][j] are a tuple 100x256
    
    'X_u_test = [X[i],T[i]] [25600,2] for interpolation (location)'
    X_u_test = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    
    'Doman bounds, for using Latin Hypercube Sampling'
    lb = X_u_test.min(0)            # [-1, 0]
    ub = X_u_test.max(0)            # [1, 0.99]
    
    'Exact Value'
    u_true = usol.flatten('F')[:,None]              # makes 2 arrays X and T such that u(X[i],T[j])=usol[i][j] are a tuple
    
    'Boundary and Initial Points'
    #Initial Condition -1 =< x =<1 and t = 0 
    leftedge_x = np.hstack((X[0,:][:,None], T[0,:][:,None])) 
    leftedge_u = usol[:,0][:,None]

    #Boundary Condition x = -1 and 0 =< t =<1
    bottomedge_x = np.hstack((X[:,0][:,None], T[:,0][:,None])) 
    bottomedge_u = usol[-1,:][:,None]

    #Boundary Condition x = 1 and 0 =< t =<1
    topedge_x = np.hstack((X[:,-1][:,None], T[:,0][:,None]))
    topedge_u = usol[0,:][:,None]

    all_X_u_train = np.vstack([leftedge_x, bottomedge_x, topedge_x]) # X_u_train [456,2] (456 = 256(L1)+100(L2)+100(L3))
    all_u_train = np.vstack([leftedge_u, bottomedge_u, topedge_u])   #corresponding u [456x1]

    #choose random N_u points for training
    idx = np.random.choice(all_X_u_train.shape[0], N_u, replace=False) 

    X_u_train = all_X_u_train[idx, :] #choose indices from  set 'idx' (x,t)
    u_train = all_u_train[idx,:]      #choose corresponding u

    'Collocation Points'
    # Latin Hypercube sampling for collocation points 
    # N_f sets of tuples(x,t)
    X_f_train = lb + (ub-lb)*lhs(2,N_f)           # rescale back to [-1, 1] and [0, 0.99]
    X_f_train = np.vstack((X_f_train, X_u_train)) # append training points to collocation points
    
    'Training data set'
    X_f_train = torch.from_numpy(X_f_train).float().to(device)
    X_u_train = torch.from_numpy(X_u_train).float().to(device)
    u_train   = torch.from_numpy(u_train).float().to(device)
    
    'Test data set'
    X_u_test  = torch.from_numpy(X_u_test).float().to(device)
    u_true    = torch.from_numpy(u_true).float().to(device)
    
    return X_f_train, X_u_train, u_train, X_u_test, u_true

