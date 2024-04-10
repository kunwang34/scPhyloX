import sys
sys.path.append('../')
from scPhyloX.mutation_rate_est import *
from scPhyloX.data_factory import *
from scPhyloX.tissue_model import *
from scPhyloX.utils import *
import numpy as np
import arviz as az
from tqdm import tqdm
import pandas as pd
from Bio import Phylo
import pickle
import argparse

parser = argparse.ArgumentParser(description='file name')
parser.add_argument('-f', type=str)
parser.add_argument('-t', type=str)
file = parser.parse_args().f
T = int(parser.parse_args().t)

tree = Phylo.read(file, format='newick')

mutnum = []
tree_dep = tree.depths() 
for i in tree.get_terminals():
    mutnum.append(tree_dep[i])
    
ge = GenerationEst(mutnum, 17)
gen_num = ge.estimate(1e5)

def para_inference_DE(data, T=20, c0=None, sigma=1, n_iter=100, bootstrape=0, verbose='text'):
    if c0 is None:
        def loss(theta):
            ax, bx, r, d, k, t0, c0 = theta
            # d = 10**(-d)
            return -my_loglike((ax, bx, r, d, k, t0), data, (T, c0, sigma))
    else:
        def loss(theta):
            ax, bx, r, d, k, t0 = theta
            # d = 10**(-d)
            return -my_loglike((ax, bx, r, d, k, t0), data, (T, c0, sigma)) 
    
    def run1():
        constraint_ueq = [lambda x: x[0]+x[1]-2]
        if c0 is None:
            de = DE(func=loss, n_dim=7, max_iter=n_iter, lb=[0, 0.995, 0, 0, 0, 0, 1], ub=[2, 1.005, 2, 0.6, 3, 20, 2000], constraint_ueq=constraint_ueq)
        else:
            de = DE(func=loss, n_dim=6, max_iter=n_iter, lb=[0, 0, 0, 1, 0, 0], ub=[2, 2, 2 ,5, 3, 20], constraint_ueq=constraint_ueq)
        bestx, besty = [], []
        for i in range(n_iter):
            xt,yt = de.run(1)
            bestx.append(xt)
            besty.append(yt)
            if verbose == 'text':
                print(f'\riter{i}, loss:{yt}, est={xt}',end = "")
            elif verbose == 'warning':
                warnings.warn(f'\riter{i}, loss:{yt}, est={xt}', Warning)
        return bestx, besty, de
    if bootstrape:
        ress_x, ress_y, des = [], [], []
        for _ in range(bootstrape):
            res_x, res_y, de = run1()
            ress_x.append(res_x)
            ress_y.append(res_y)
            des.append(de) 
        return np.array(ress_x), np.array(ress_y), des
    else:
        
        return run1()

res = para_inference_DE(gen_num, T=T, sigma=10000, n_iter=150, bootstrape=0)
pickle.dump(res, open(f"/home/wangkun/phylodynamics_new/results/hsc/{file.split('/')[-1].split('_')[1]}_de.pkl", 'wb'))
axh, bxh, rh, dh, kh, t0h, c0 = res[0][-1]
res_de = (axh, bxh, rh, dh, kh, t0h)

logl = LogLike(my_loglike, gen_num, (T, c0, 10000))
with pm.Model() as model:
    ax = pm.TruncatedNormal('ax', mu=axh, sigma=0.1, lower=0, upper=2, initval=axh)
    bx = pm.TruncatedNormal('bx', mu=bxh, sigma=0.1, lower=0, upper=2-ax, initval=bxh)
    r = pm.TruncatedNormal('r', mu=rh, sigma=0.1, lower=0.1, initval=rh)
    k = pm.TruncatedNormal('k', mu=kh, sigma=0.1, lower=0, upper=3, initval=kh)
    t0 = pm.TruncatedNormal('t0', mu=t0h, sigma=0.3, lower=0, upper=20, initval=t0h)
    d = pm.Beta('d', alpha=1, beta=1/dh-1, initval=dh)
    theta = pt.as_tensor_variable([ax, bx, r, d, k, t0])

    # use a Potential to "call" the Op and include it in the logp computation
    pm.Potential("likelihood", logl(theta))
    idata = pm.sample(1000, tune=1000, step=pm.DEMetropolis(), chains=7)

pickle.dump(idata, open(f"/home/wangkun/phylodynamics_new/results/hsc/{file.split('/')[-1].split('_')[1]}.pkl", 'wb'))