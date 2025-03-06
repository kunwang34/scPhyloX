import sys
sys.path.append('../')
import scPhyloX as spx
import numpy as np
import arviz as az
from tqdm import tqdm
import pandas as pd
from Bio import Phylo
import pickle
import argparse
from sko.DE import DE
from scipy.integrate import quad, solve_ivp
import pymc as pm
import pytensor
import pytensor.tensor as pt

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
    
ge = spx.est_mr.GenerationEst(mutnum, 17)
gen_num = ge.estimate(1e5)

def para_inference_DE(data, T=20, c0=None, sigma=1, n_iter=100, bootstrape=0, verbose='text'):
    '''
    Mutation rate estimation using DE
    
    Args:
        data:
            lp-dist
        n_iter:
            Iterations of de estimation
        bootstrape:
            Weather using bootstrape to accuratly estimate mutation rate, 0 to turn off.
    Return:
        tuple:
            (accepted parameters, loss, de-estimator)
    '''
    if c0 is None:
        def loss(theta):
            ax, bx, r, k, t0, r1, b1, c0 = theta
            return -spx.est_haem.my_loglike((ax, bx, r, k, t0, r1, b1), data, (T, c0, sigma))
    else:
        def loss(theta):
            ax, bx, r, k, t0, r1, b1 = theta

            return -spx.est_haem.my_loglike((ax, bx, r, k, t0, r1, b1), data, (T, c0, sigma)) 
    
    def run1():
        constraint_ueq = [lambda x: x[0]+x[1]-2]
        if c0 is None:
            de = DE(func=loss, n_dim=8, max_iter=n_iter, lb=[0, 0.999, 0, 1, 0, 0, 0, 1], ub=[2, 1.001, 3, 3, T, 1, 1, 2000], constraint_ueq=constraint_ueq)
        else:
            de = DE(func=loss, n_dim=7, max_iter=n_iter, lb=[0, 0, 0, 1, 0, 0, 0], ub=[2, 2, 3, 5, 20, 1, 1], constraint_ueq=constraint_ueq)
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
    
    

res = para_inference_DE(gen_num, T=T, sigma=1000, n_iter=50, bootstrape=0)
pickle.dump(res[:2], open(f"../results/hsc/{file.split('/')[-1].split('_')[1]}_de.pkl", 'wb'))
axh, bxh, rh, kh, t0h, r1h, b1h, c0 = res[0][-1]

logl = spx.est_haem.LogLike(spx.est_haem.my_loglike, gen_num, (T, c0, 1000))
with pm.Model() as model:
    ax = pm.TruncatedNormal('ax', mu=axh, sigma=0.1, lower=0, upper=2, initval=axh)
    # bx = pm.TruncatedNormal('bx', mu=bxh, sigma=0.1, lower=0, upper=2, initval=bxh)
    r = pm.TruncatedNormal('r', mu=rh, sigma=0.1, lower=0.1, initval=rh)
    k = pm.TruncatedNormal('k', mu=kh, sigma=0.1, lower=0, upper=5, initval=kh)
    t0 = pm.TruncatedNormal('t0', mu=t0h, sigma=0.3, lower=0, upper=20, initval=t0h)
    r1 = pm.TruncatedNormal('r1', mu=r1h, sigma=0.1, lower=0.1, initval=r1h)
    b1 = pm.Beta('b1', alpha=1, beta=1/b1h-1, initval=b1h)

    theta = pt.as_tensor_variable([ax, 1, r, k, t0, r1, b1])
    pm.Potential("likelihood", logl(theta))
    idata = pm.sample(1000, tune=1000, step=pm.DEMetropolis(), chains=8)


pickle.dump(idata, open(f"../results/hsc/{file.split('/')[-1].split('_')[1]}.pkl", 'wb'))