import torch
import numpy as np
from scipy.special import erf, gamma
import normflows as nf
#from nf import distributions
class Sharp(nf.distributions.Target):
    def __init__(self):
        super().__init__(prop_scale=torch.tensor(1.0), 
                         prop_shift=torch.tensor(0.0))
        self.ndims = 2
        self.targetval = 4.0
    def prob(self,x):
        return -torch.log(x[:,0])/torch.sqrt(x[:,0])
    def log_prob(self, x):
        return torch.log(torch.abs(self.prob(x)))
    
class Gauss(nf.distributions.Target):
    def __init__(self, ndims=2, alpha=0.2):
        super().__init__(prop_scale=torch.tensor(1.0), 
                         prop_shift=torch.tensor(0.0))
        self.ndims = ndims
        self.alpha = alpha
        self.log_const = -self.ndims *(np.log(self.alpha) + 0.5 * np.log(np.pi))
        self.targetval = erf(1/(2.*self.alpha))**self.ndims
    def log_prob(self, x):
        return -1.0 * torch.sum((x-0.5)**2/self.alpha**2, -1) + self.log_const
    def prob(self, x):
        return torch.exp(self.log_prob(x))
    
class Camel(nf.distributions.Target):
    #Target value not implemented
    def __init__(self, ndims=2, alpha = 0.2, pos = 1./3.):
        super().__init__(prop_scale=torch.tensor(1.0), 
                         prop_shift=torch.tensor(0.0))
        self.ndims = ndims
        self.alpha = alpha
        self.pos = pos
        self.pre1 = np.exp(-self.ndims *(np.log(self.alpha) + 0.5 * np.log(np.pi)))
        self.pre2 = np.exp(-self.ndims *(np.log(self.alpha) + 0.5 * np.log(np.pi)))
        #self.targetval =  0.5* (0.5*(erf(1/(3.*alpha))+erf(2/(3.*alpha))))**ndims + 0.1/16.0* (0.5*(erf(1/(3.*alpha/4.0))+erf(2/(3.*alpha/4.0))))**ndims 
    def log_prob(self, x):
        return torch.log(self.prob(x))
    def prob(self, x):
        exp1 = -1.0 * torch.sum((x-(self.pos))**2/self.alpha**2, -1)
        exp2 = -1.0 * torch.sum((x-(1. - self.pos))**2/self.alpha**2, -1) 
        return 0.5 * (self.pre1 * torch.exp(exp1) +  self.pre2 * torch.exp(exp2)) 

class Sphere(nf.distributions.Target):
    def __init__(self, ndims=2):
        super().__init__(prop_scale=torch.tensor(1.0), 
                         prop_shift=torch.tensor(0.0))
        self.ndims =ndims
        self.log_const = -self.ndims *(np.log(self.alpha) + 0.5 * np.log(np.pi))
        self.targetval = erf(1/(2.*self.alpha))**self.ndims
    def log_prob(self, x):
        return -1.0 * torch.sum((x-0.5)**2/self.alpha**2, -1) + self.log_const
    def prob(self, x):
        return torch.exp(self.log_prob(x))

