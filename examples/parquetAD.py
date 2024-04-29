import os
import pandas as pd
import numpy as np
import torch
import re
import normflows as nf
from nsf_integrator import generate_model, train_model 
from funcs_sigma import *
#from matplotlib import pyplot as plt
import tracemalloc
#from torch.utils.viz._cycles import warn_tensor_cycles

#warn_tensor_cycles()

root_dir = os.path.join(os.path.dirname(__file__), "source_codeParquetAD/")
#include(os.path.join(root_dir, f"func_sigma_o100.py"))
#from absl import app, flags

       



def _StringtoIntVector(s):
    pattern = r"[-+]?\d+"
    return [int(match) for match in re.findall(pattern, s)]


class FeynmanDiagram(nf.distributions.Target):
    @torch.no_grad()
    def __init__( self, loopBasis, leafstates, leafvalues, batchsize):
        super().__init__( prop_scale=torch.tensor(1.0), 
                          prop_shift=torch.tensor(0.0))
        lftype, lforders, leaf_tau_i, leaf_tau_o, leafMomIdx = leafstates
        #leaf states information
        self.register_buffer("lftype", lftype)
        self.register_buffer("lforders", lforders)
        self.register_buffer("leaf_tau_i", leaf_tau_i)
        self.register_buffer("leaf_tau_o", leaf_tau_o)
        self.register_buffer("leafMomIdx", leafMomIdx) 


        self.eps0 = 1 / (4*np.pi)
        self.e0 = np.sqrt(2)
        self.mass2 = 0.5
        self.me = 0.5
        self.spin = 2
        self.rs = 2.0
        self.dim = 3
        self.kF = (9*np.pi / (2*self.spin))**(1 / 3) / self.rs 
        self.EF = self.kF**2 / (2*self.me)
        self.mu = self.EF
        self.beta = 50.0/self.EF
        print("param:",  self.dim, self.beta, self.me, self.mass2, self.mu , self.e0, self.eps0)
        self.extk = self.kF
        self.extn = 0

        self.innerLoopNum = 2
        self.totalTauNum = 2

        self.register_buffer("loopBasis", loopBasis)
        self.register_buffer("loops", torch.empty((self.dim, self.loopBasis.shape[0])))
        #self.loops = torch.empty((self.dim, self.loopBasis.shape[0]))
        self.targetval = 4.0
        self.batchsize = batchsize
        self.ndims = self.innerLoopNum*self.dim + self.totalTauNum - 1

        self.maxK = 10*self.kF

        self.register_buffer("leafvalues", torch.broadcast_to(leafvalues, (self.batchsize, leafvalues.shape[0])))
        self.register_buffer("p", torch.zeros([self.batchsize, self.dim, self.innerLoopNum + 1]))
        self.register_buffer("tau",torch.zeros(self.leafvalues.shape) )
        self.register_buffer("kq2", torch.zeros(self.leafvalues.shape))
        self.register_buffer("inK", torch.zeros(self.leafvalues.shape))
        self.register_buffer("dispersion", torch.zeros(self.leafvalues.shape))
        self.register_buffer("factor",  torch.ones([self.batchsize,]) )
        self.register_buffer("root",  torch.ones([self.batchsize,]) )
        self.register_buffer("isfermi",  torch.full(self.leafvalues.shape, True) )
        self.register_buffer("isbose",  torch.full(self.leafvalues.shape, True) )
        self.register_buffer("leaf_fermi", torch.zeros(self.leafvalues.shape))
        self.register_buffer("leaf_bose", torch.zeros(self.leafvalues.shape))
        
        self.register_buffer("samples", torch.zeros([self.batchsize, self.ndims]))
        self.register_buffer("log_q", torch.zeros([self.batchsize]))
        self.register_buffer("log_det", torch.zeros([self.batchsize]))
        self.register_buffer("val", torch.zeros([self.batchsize]))

        self.p[:,0,0] += self.kF
        # self.p[:,:,1] += torch.tensor([-0.15634164444323098, 0.3807986718804936, 0.27070297306504326])
        # self.p[:,:,2] += torch.tensor([ -0.03784388142908468, 0.2999292168299123, 0.8180187845829805])
       
        # self.leafstates = leafstates
        # self.leafvalues = torch.broadcast_to(leafvalues, (self.batchsize, leafvalues.shape[0]))
        
        # self.p = torch.zeros([self.batchsize, self.dim, self.innerLoopNum + 1])
        # self.p[:,0,0] += self.kF
        # self.tau = torch.zeros([self.batchsize, self.leafvalues.shape[-1]])
        # self.kq2 = torch.zeros([self.batchsize, self.leafvalues.shape[-1]])
        # self.inK = torch.zeros([self.batchsize, self.leafvalues.shape[-1]])
        # self.dispersion = torch.zeros([self.batchsize, self.leafvalues.shape[-1]])
        # self.factor = torch.ones([self.batchsize,]) 
        # self.root = torch.ones([self.batchsize,]) 
        # self.isfermi = torch.full(self.leafvalues.shape, True) 
        # self.isbose = torch.full(self.leafvalues.shape, True)
        # self.leaf_fermi = torch.zeros([self.batchsize, self.leafvalues.shape[-1]])
        # self.leaf_bose = torch.zeros([self.batchsize, self.leafvalues.shape[-1]])
        #Convention of variables: first totalTauNum - 1 variables are tau. The rest are momentums in shperical coordinate.

    @torch.no_grad()
    def kernelFermiT(self):
        sign = torch.where(self.tau > 0, 1.0, -1.0)

        a = torch.where(self.tau > 0, torch.where( self.dispersion > 0, -self.tau, self.beta - self.tau), torch.where( self.dispersion > 0, -(self.beta + self.tau) , -self.tau) )
        b = torch.where( self.dispersion > 0, -self.beta, self.beta)

        # Use torch operations to ensure calculations are done on GPU if tensors are on GPU
        self.leaf_fermi = sign * torch.exp(self.dispersion * a)                                        
        self.leaf_fermi /= (1 + torch.exp(self.dispersion * b))
        #return leaf_fermi

    @torch.no_grad()
    def extract_mom(self, var):
        p_rescale = var[:, self.totalTauNum-1 : self.totalTauNum-1 + self.innerLoopNum]
        theta = var[:, self.totalTauNum-1 + self.innerLoopNum : self.totalTauNum-1 + 2*self.innerLoopNum]*np.pi
        phi = var[:, self.totalTauNum-1 + 2*self.innerLoopNum : self.totalTauNum-1 + 3*self.innerLoopNum]*2*np.pi
        #print((p_rescale / (1+1e-6 - p_rescale)**2)**2 * torch.sin(theta))
        #self.factor = torch.prod((p_rescale / (1+1e-6 - p_rescale)**2)**2 * torch.sin(theta), dim = 1) 
        #print("factor:", self.factor)
        #p_rescale /= (1.0 + 1e-10 - p_rescale)

        p_rescale *= self.maxK
        self.factor = torch.prod(p_rescale**2 * torch.sin(theta), dim = 1) 
        self.p[:, 0, 1:] = p_rescale * torch.sin(theta) 
        self.p[:, 1, 1:] = self.p[:, 0, 1:]
        self.p[:, 0, 1:] *= torch.cos(phi)
        self.p[:, 1, 1:] *= torch.sin(phi)
        self.p[:, 2, 1:] = p_rescale * torch.cos(theta)
       
    @torch.no_grad()   
    def _evalleaf(self,var):
        self.isfermi[:] = (self.lftype == 1)
        self.isbose[:] = (self.lftype == 2)
        #update momentum
        self.extract_mom(var) #varK should have shape [batchsize, dim, innerLoopMom]
        #print(self.p) #, "\n", var[:, self.totalTauNum-1 : self.totalTauNum-1 + self.innerLoopNum])
        self.loops = torch.tensordot(self.p, self.loopBasis, dims = ([-1], [0])) #loopBasis has shape [innerLoopMom, ]
        #print(torch.tensordot(self.p, self.loopBasis[:, :2], dims = ([-1], [0])))
        #Calculate fermionic leaves
        self.tau = torch.where(self.leaf_tau_o==0, 0.0, var[:, self.leaf_tau_o-1])
        self.tau -= torch.where(self.leaf_tau_i==0, 0.0, var[:, self.leaf_tau_i-1])
        self.tau *= self.beta       
        
        kq = self.loops[:, :, self.leafMomIdx]
        #print("test1:", lftype, self.p, self.loops.shape, self.loopBasis[:, :2], kq)
        self.kq2 = torch.sum(kq*kq, dim = 1)
        self.dispersion = self.kq2 / (2 * self.me) - self.mu
        #print("test2:",self.loops.shape, eps.shape, tau.shape, leaf_tau_i.shape)
        #order = lforders[0]
        self.kernelFermiT()
        #print("var", kq2, self.mu, kernelFermiT(tau, eps, self.beta), tau, eps, self.beta)
        #Calculate bosonic leaves
        #order = lforders[1]
        self.invK = 1.0 / (self.kq2 + self.mass2)
        self.leaf_bose = ((self.e0 ** 2 / self.eps0) * self.invK)
        self.leaf_bose *= (self.mass2 * self.invK) ** self.lforders[1]
        self.leafvalues = torch.where(self.isfermi,  self.leaf_fermi, self.leafvalues)
        self.leafvalues = torch.where(self.isbose,  self.leaf_bose, self.leafvalues)
        #print(self.leafvalues)

    @torch.no_grad()
    def prob(self, var):
        self._evalleaf(var)
        self.root = torch.stack(func_sigma_o200.graphfunc(self.leafvalues), dim=0).sum(dim=0) * self.factor * (self.maxK * 2*np.pi**2)**(self.innerLoopNum)*(self.beta)**(self.totalTauNum-1)/ (2*np.pi)**(self.dim * self.innerLoopNum)
        #self.root = (func_sigma_o100.graphfunc(self.leafvalues) * self.factor * (self.maxK * 2*np.pi**2)**(self.innerLoopNum)/ (2*np.pi)**(self.dim * self.innerLoopNum)).detach()
        #print("fermi",self.leaf_fermi, "tau", self.tau, "root", self.root)
        return self.root
    
    @torch.no_grad()
    def log_prob(self, var):
        self.prob(var)
        return torch.log(torch.where(torch.abs(self.root)>1e-10, torch.abs(self.root), torch.abs(self.root) + 1e-10))


def main(argv):
    del argv    
   
    # partition, diagpara, extT_labels = diagram_info
    partition = [(2,0,0)]
    maxMomNum = max(key[0] for key in partition) + 1
    name="sigma"
    df = pd.read_csv(os.path.join(root_dir, f"loopBasis_{name}_maxOrder6.csv"))
    #print([df[col].tolist() for col in df.columns])
    loopBasis = torch.tensor([df[col].iloc[:maxMomNum].tolist() for col in df.columns]).T
    leafstates = []
    leafvalues = []
    
    for key in partition:
        key_str = ''.join(map(str, key))
        df = pd.read_csv(os.path.join(root_dir, f"leafinfo_{name}_{key_str}.csv"))
        # leafstates_par = [LeafStateAD(row[1], _StringtoIntVector(row[2]), *row[3:]) for index, row in df.iterrows()]
        # #print(leafstates_par[1].orders)
        # leafstates.append(leafstates_par)
        # leafvalues.append(df[df.columns[0]].tolist())
        leaftypes = []
        leaforders =[]
        inTau_idx = []
        outTau_idx = []
        loop_idx = []
        for index, row in df.iterrows():
            leaftypes.append(torch.tensor(row[1])) 
            leaforders.append(torch.tensor(_StringtoIntVector(row[2])))
            inTau_idx.append(torch.tensor(row[3]))
            outTau_idx.append(torch.tensor(row[4]))
            loop_idx.append(torch.tensor(row[5]))

        leaftypes = torch.stack(leaftypes)
        leaforders = torch.stack(leaforders).T
        inTau_idx = torch.stack(inTau_idx) - 1 
        outTau_idx = torch.stack(outTau_idx) - 1 
        loop_idx = torch.stack(loop_idx) - 1 
        #print( leaftypes == 1,leaforders.shape,inTau_idx, outTau_idx, loop_idx)
        leafstates = (leaftypes, leaforders,inTau_idx, outTau_idx, loop_idx)
        # print(leafstates_par[1].orders)
        # leafstates.append(leafstates_par)
        leafvalues = torch.tensor(df[df.columns[0]].tolist())

   
  
    
    diagram = FeynmanDiagram(loopBasis, leafstates, leafvalues, 10000)
    # q0 = nf.distributions.base.Uniform(diagram.ndims, 0.0, 1.0)
    # samples = 0.0 * q0.sample(diagram.batchsize)
    # samples[:, 0] += 50.45739393425352/diagram.beta
    # diagram.prob(samples)
    # diagram.prob(samples)
    #print(func_sigma_o300.graphfunc(leafvalues))
    

        
    
   
   
   
    nfm = generate_model(diagram)   
    epochs = 100
    # torch.cuda.memory._record_memory_history()
    tracemalloc.start()
    blocks = 10
    with torch.no_grad():
        mean, err = nfm.integrate_block(blocks)
    
    print("Result with {:d} is {:.5e} +/- {:.5e}. \n Target result:{:.5e}".format(
            blocks*diagram.batchsize,  mean, err, nfm.p.targetval))
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    print("[ Top 20 ]")
    for stat in top_stats[:20]:
        print(stat)
    
    train_model(nfm, epochs, diagram.batchsize)
    with torch.no_grad():
        mean, err = nfm.integrate_block(blocks)
    # nfm.train()
    print("Result with {:d} is {:.5e} +/- {:.5e}. \n Target result:{:.5e}".format(
            blocks*diagram.batchsize,  mean, err, nfm.p.targetval))
    # torch.cuda.memory._dump_snapshot("my_snapshot.pickle")

if __name__ == '__main__':
    main(1)
    #app.run(main)