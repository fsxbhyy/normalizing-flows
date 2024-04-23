import numpy as np
from math import sqrt, tanh, pi

def integrandKW_Clib(idx, vars, config):
    varK, varT, varN, ExtKidx = vars
    para, kgrid, ngrid, MaxLoopNum, extT_labels = config['userdata'][:5]
    leafstates, leafval = config['userdata'][6][idx], config['userdata'][7][idx]
    momLoopPool, root = config['userdata'][8:10]
    isLayered2D = config['userdata'][10]
    partition = config['userdata'][11]

    dim, beta, me, lambd, mu, e0, eps0 = para['dim'], para['beta'], para['me'], para['mass2'], para['mu'], para['e0'], para['eps0']
    extidx = ExtKidx[0]
    varK.data[0, 0] = kgrid[extidx]

    momLoopPool.update(varK.data[:, :MaxLoopNum])
    for i, lfstat in enumerate(leafstates):
        lftype, lforders, leaf_tau_i, leaf_tau_o, leafMomIdx = lfstat['type'], lfstat['orders'], lfstat['inTau_idx'], lfstat['outTau_idx'], lfstat['loop_idx']
        if lftype == 0:
            continue
        elif lftype == 1:  # fermionic
            tau = varT[leaf_tau_o] - varT[leaf_tau_i]
            kq = momLoopPool.loop(leafMomIdx)
            eps = np.dot(kq, kq) / (2 * me) - mu
            order = lforders[0]
            leafval[i] = Propagator.green_derive(tau, eps, beta, order)
        elif lftype == 2:  # bosonic
            kq = momLoopPool.loop(leafMomIdx)
            order = lforders[1]
            if dim == 3:
                invK = 1.0 / (np.dot(kq, kq) + lambd)
                leafval[i] = (e0 ** 2 / eps0) * invK * (lambd * invK) ** order
            elif dim == 2:
                if not isLayered2D:
                    invK = 1.0 / (sqrt(np.dot(kq, kq)) + lambd)
                    leafval[i] = (e0 ** 2 / (2 * eps0)) * invK * (lambd * invK) ** order
                else:
                    if order == 0:
                        q = sqrt(np.dot(kq, kq) + 1e-16)
                        invK = 1.0 / q
                        leafval[i] = (e0 ** 2 / (2 * eps0)) * invK * tanh(lambd * q)
                    else:
                        leafval[i] = 0.0  # no high-order counterterms
            else:
                raise Exception("not implemented!")
        else:
            raise Exception(f"this leaftype {lftype} not implemented!")

    group = partition[idx]
    evalfuncParquetAD_sigma_map[group](root, leafval)

    n = ngrid[varN[0]]
    weight = sum(root[i] * phase(varT, extT, n, beta) for i, extT in enumerate(extT_labels[idx]))
    loopNum = config['dof'][idx][0]
    factor = 1.0 / (2 * pi) ** (dim * loopNum)
    return weight * factor

# Placeholder functions for undefined parts
def phase(varT, extT, n, beta):
    # Dummy implementation
    return np.exp(-n * beta * (varT - extT))
