import os
os.chdir('..')

import scPhyloX as spx
from scipy.integrate import solve_ivp
import numpy as np
from scipy.stats import poisson
import pickle
import gzip
from time import time
import argparse
import arviz as az 

parser = argparse.ArgumentParser(description='file name')
parser.add_argument('-f', type=str)
i = parser.parse_args().f

with open('/home/wangkun/scPhyloX/datasets/cellnumber_sample/computational_times.txt', 'a') as f:
    data = pickle.load(gzip.open(f'/home/wangkun/scPhyloX/datasets/cellnumber_sample/{i}', 'rb'))
    seqtab_full = np.array([i.seq for i in data['SC']] + [i.seq for i in data['DC']])
    seqtab_full = seqtab_full[np.random.choice(range(seqtab_full.shape[0]), 500, replace=False)]
    for sr in [1, 0.7, 0.5, 0.2]:
        seqtab = seqtab_full[np.random.choice(range(500), int(sr*500), replace=False)]
        time0 = time()
        branch_len = spx.data_factory.get_branchlen(seqtab)
        mutnum = spx.data_factory.get_mutnum(seqtab)
        idata_bl = spx.est_mr.mutation_rate_mcmc(branch_len, draw=500, tune=500)
        time1 = time()
        pickle.dump(idata_bl, gzip.open(f"/home/wangkun/scPhyloX/datasets/cellnumber_sample/{i.replace('.pkl.gz', f'_idata_bl_{sr}.pkl')}", 'wb'))
        time2 = time()
        ge = spx.est_mr.GenerationEst(mutnum, az.summary(idata_bl)['mu'])
        gen_num = ge.estimate(data['cell_num'][-1].sum())
        for formula in i.split('_')[:-1]:
            exec(formula)
        ax = a*(1+p)
        bx = b*(1+p)+1-p
        axh, bxh, rh, dh, kh, t0h, dh = (ax,bx,r,d,k,t0, d)
        mcmc_prior = (axh, bxh, rh, dh, kh, t0h)
        idata = spx.est_tissue.mcmc_inference(gen_num, mcmc_prior, T=35, c0=100, sigma=100)
        time3 = time()
        pickle.dump(idata, gzip.open(f"/home/wangkun/scPhyloX/datasets/cellnumber_sample/{i.replace('.pkl.gz', f'_idata_mn_{sr}.pkl')}", 'wb'))
        f.write(f"{i.split('_')[-1]}\t{int(sr*500)}\t{time1-time0}\t{time3-time2}\n")
