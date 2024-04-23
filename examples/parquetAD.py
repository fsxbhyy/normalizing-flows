import os
import pandas as pd
import numpy as np
import torch
import re
import normflows as nf
from nsf_integrator import generate_model, train_model 
from funcs_sigma import *


root_dir = os.path.join(os.path.dirname(__file__), "source_codeParquetAD/")
#include(os.path.join(root_dir, f"func_sigma_o100.py"))
#from absl import app, flags

class LeafStateAD:
    def __init__(self, type, orders, inTau_idx, outTau_idx, loop_idx):
        self.type = type
        self.orders = orders
        self.inTau_idx = inTau_idx
        self.outTau_idx = outTau_idx
        self.loop_idx = loop_idx

#from your_module.leaf_state_ad import LeafStateAD

def kernelFermiT(τ, ω, β):
    # Condition checks in PyTorch, keeping the results in tensor format
    # condition_τ_range = (-β < τ) & (τ <= β)
    # if not torch.all(condition_τ_range):
    #    raise ValueError("τ values must be in the range (-β, β]")
    sign = torch.where(τ > 0, 1.0, -1.0)

    a = torch.where(τ > 0, torch.where( ω > 0, -τ, β - τ), torch.where( ω > 0, -(β + τ) , -τ) )
    b = torch.where( ω > 0, -β, β)

    # Use torch operations to ensure calculations are done on GPU if tensors are on GPU
    exp_ωa = torch.exp(ω * a)
    exp_ωb = torch.exp(ω * b)
    result = sign * exp_ωa / (1 + exp_ωb)

    return result

def _StringtoIntVector(s):
    pattern = r"[-+]?\d+"
    return [int(match) for match in re.findall(pattern, s)]


class FeynmanDiagram(nf.distributions.Target):
    def __init__( self, loopBasis, leafstates, leafvalues):
        super().__init__( prop_scale=torch.tensor(1.0), 
                          prop_shift=torch.tensor(0.0))
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

        self.innerLoopNum = 1
        self.totalTauNum = 1

        self.loopBasis = loopBasis
        self.loops = torch.empty((self.dim, self.loopBasis.shape[0]))

        self.batchsize = 10000
        self.ndims = self.innerLoopNum*self.dim + self.totalTauNum - 1

        self.maxK = 30*self.kF

        self.leafstates = leafstates
        self.leafvalues = torch.broadcast_to(leafvalues, (self.batchsize, leafvalues.shape[0]))
        self.if_cut_mom = False
        self.targetval = 4.0
        self.Cartesian = False
        
        self.p = torch.zeros([self.batchsize, self.dim, self.innerLoopNum + 1])
        self.p[:,0,0] += self.kF
        #self.p[:,:,1] += 0.554013278447432
        self.p[:,:,1] += torch.tensor([-0.7343461203665687, 0.5154819821950148, -0.28596976571599436])
        self.factor = torch.ones([self.batchsize,]) 
        self.root = torch.ones([self.batchsize,]) 
        #Convention of variables: first totalTauNum - 1 variables are tau. The rest are momentums in shperical coordinate.
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
        self.p[:, 0, 1:] = p_rescale * torch.sin(theta) * torch.cos(phi)
        self.p[:, 1, 1:] = p_rescale * torch.sin(theta) * torch.sin(phi)
        self.p[:, 2, 1:] = p_rescale * torch.cos(theta)

    def _evalleaf(self,var):
        lftype, lforders, leaf_tau_i, leaf_tau_o, leafMomIdx = self.leafstates.type, self.leafstates.orders, self.leafstates.inTau_idx, self.leafstates.outTau_idx, self.leafstates.loop_idx
        isfermi = torch.broadcast_to(lftype == 1, self.leafvalues.shape)
        isbose = torch.broadcast_to(lftype == 2, self.leafvalues.shape)
        
        #update momentum
        self.extract_mom(var) #varK should have shape [batchsize, dim, innerLoopMom]
        
        #print(self.p) #, "\n", var[:, self.totalTauNum-1 : self.totalTauNum-1 + self.innerLoopNum])
        self.loops = torch.tensordot(self.p, self.loopBasis, dims = ([-1], [0])) #loopBasis has shape [innerLoopMom, ]
        #print(torch.tensordot(self.p, self.loopBasis[:, :2], dims = ([-1], [0])))
        #Calculate fermionic leaves
        tau = (var[:, leaf_tau_o] - var[:, leaf_tau_i]) * self.beta
        #print(tau)
        kq = self.loops[:, :, leafMomIdx]
        #print("test1:", lftype, self.p, self.loops.shape, self.loopBasis[:, :2], kq)
        kq2 = torch.sum(kq*kq, dim = 1)
        eps = kq2 / (2 * self.me) - self.mu
        #print("test2:",self.loops.shape, eps.shape, tau.shape, leaf_tau_i.shape)
        order = lforders[0]
        leaf_fermi = kernelFermiT(tau, eps, self.beta)
        #print("var", kq2, self.mu, kernelFermiT(tau, eps, self.beta), tau, eps, self.beta)
        #Calculate bosonic leaves
        order = lforders[1]
        invK = 1.0 / (kq2 + self.mass2)
        leaf_bose = ((self.e0 ** 2 / self.eps0) * invK * (self.mass2 * invK) ** order)
        self.leafvalues = torch.where(isfermi,  leaf_fermi, torch.where(isbose, leaf_bose, self.leafvalues))
        #print(self.leafvalues)

    def prob(self, var):
        self._evalleaf(var)
        self.root = (func_sigma_o100.graphfunc(self.leafvalues) * self.factor * (self.maxK * 2*np.pi**2)**(self.innerLoopNum)/ (2*np.pi)**(self.dim * self.innerLoopNum)).detach()
        #print(self.p)
        return self.root
    def log_prob(self, var):
        self.prob(var)
        return torch.log(torch.where(torch.abs(self.root)>1e-10, torch.abs(self.root), torch.abs(self.root) + 1e-10))


def main(argv):
    del argv    
   
    # partition, diagpara, extT_labels = diagram_info
    partition = [(1,0,0)]
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
        leafstates = LeafStateAD(leaftypes,leaforders,inTau_idx, outTau_idx, loop_idx)
        # print(leafstates_par[1].orders)
        # leafstates.append(leafstates_par)
        leafvalues = torch.tensor(df[df.columns[0]].tolist())

   

    print(leafvalues)#, leafstates)
    diagram = FeynmanDiagram(loopBasis, leafstates, leafvalues)
    #samples = q0.sample(diagram.batchsize)
    #print(diagram.prob(samples))
    #print(func_sigma_o300.graphfunc(leafvalues))

   
    #print(leafstates, leafvalues)
    # root = np.zeros(max(len(labels) for labels in extT_labels))
  
    # vardim = dim * momnum + taunum

    # ndims = FLAGS.ndims
    # alpha = FLAGS.alpha
    # nsamples = FLAGS.nsamples
    # epochs = FLAGS.epochs
    # target = integrandKW()
    
    nfm = generate_model(diagram)   
    epochs = 100

    nfm.eval()
    blocks = 100
    block_samples = diagram.batchsize
    mean, err = nfm.integrate_block(block_samples, blocks)
    nfm.train()
    print("Result with {:d} is {:.5e} +/- {:.5e}. \n Target result:{:.5e}".format(
            blocks*block_samples,  mean, err, nfm.p.targetval))

    train_model(nfm, epochs, diagram.batchsize)
    
    mean, err = nfm.integrate_block(block_samples, blocks)
    print("Result with {:d} is {:.5e} +/- {:.5e}. \n Target result:{:.5e}".format(
            blocks*block_samples,  mean, err, nfm.p.targetval))


if __name__ == '__main__':
    main(1)
    #app.run(main)