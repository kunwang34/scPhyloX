import sys
sys.path.append('../')
from scPhyloX.mutation_rate_est import *
from scPhyloX.data_factory import *
from scPhyloX.tumor_model import *
from scPhyloX.utils import *
import numpy as np
import arviz as az
# import matplotlib.pyplot as plt
from tqdm import tqdm
# import seaborn as sns
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='file name')
parser.add_argument('-f', type=str)
parser.add_argument('-p', type=str)
file = parser.parse_args().f
path = parser.parse_args().p

data = []
name = []
with open(f'{path}/{file}', 'r')  as f:
    line = f.readline()
    line = f.readline()
    while line:
        line = f.readline()
        try:
            data.append(np.array(list(line.split(' ')[1].strip())).astype(int))
            name.append(line.split(' ')[0].strip())
            
        except:
            None
data = np.array(data)
data = np.unique(data,axis=0)
mutnum = get_mutnum(data)

ge = GenerationEst(mutnum, 0.4)
gen_num = ge.estimate(tumor_size['_'.join(file.split('_')[:2])])

T= 28

res = para_inference_DE(gen_num, T=T, sigma=10000, n_iter=100, bootstrape=0)

rh, ah, sh, uh, c0 = res[0][-1]
uh = 10**-(uh*5)
res_de = (rh, ah, sh, uh)

logl = LogLike(my_loglike, gen_num, (T, c0, 10000))
with pm.Model() as model:
    r = pm.TruncatedNormal('r', mu=rh, sigma=0.1, lower=0, upper=5, initval=rh)
    a = pm.TruncatedNormal('a', mu=ah, sigma=0.1, lower=0.5, upper=1, initval=ah)
    s = pm.TruncatedNormal('s', mu=sh, sigma=0.1, lower=0, upper=0.5, initval=sh)
    u = pm.Beta('u', alpha=1, beta=1/uh-1, initval=uh)
    theta = pt.as_tensor_variable([r, a, s, u])

    # use a Potential to "call" the Op and include it in the logp computation
    pm.Potential("likelihood", logl(theta))
    idata = pm.sample(1000, tune=1000, step=pm.DEMetropolis(), chains=5)
    
import pickle
pickle.dump(idata, open(f'../results/IBD_tumor/{file}', 'wb'))
pickle.dump(res[:2], open(f'../results/IBD_tumor/{file}_de', 'wb'))