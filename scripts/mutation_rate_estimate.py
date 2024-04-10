# import os
# os.chdir('../')
import sys
sys.path.append('../')
from scPhyloX.mutation_rate_est import *
from scPhyloX.data_factory import *
import arviz as az
import numpy as np
import argparse
import pickle
parser = argparse.ArgumentParser(description='dataset name')
parser.add_argument('-f', type=str)
parser.add_argument('-o', type=str)
fly = parser.parse_args().f
organ = parser.parse_args().o
# fly, organ = 'L5', 'Br'

mutnum = mutnum_fly(f'../datasets/FLY/pre_processed/{fly}/{organ}.txt', '../datasets/FLY/pre_processed/ref')
branchlen = branchlen_fly(f'../datasets/FLY/pre_processed/{fly}/{organ}.txt', '../datasets/FLY/pre_processed/ref')
N, sr = population_size[organ], len(mutnum)/population_size[organ]
idata = mutation_rate_mcmc(branchlen, draw=500, tune=500)
az.summary(idata).to_csv(f'../results/mutrate_{fly}_{organ}.csv')
pickle.dump(idata, open(f'../results/mutrate_{fly}_{organ}.pkl', 'wb'))