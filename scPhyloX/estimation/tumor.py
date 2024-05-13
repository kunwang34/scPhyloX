import numpy as np
from scipy.special import factorial, comb
from sympy import polylog
from scipy.stats import norm, beta
import pymc as pm
import pytensor
import pytensor.tensor as pt
import arviz as az
from scipy.integrate import quad, solve_ivp
from sko.DE import DE
from io import StringIO
from copy import deepcopy
from tqdm import tqdm
import sys
import warnings
from collections import defaultdict

def cellnumber(t, xx, r, a, s, u):
    '''
    ODE of cell number changes over time
    
    Args:
        t: 
            time
        xx: 
            [n_stemcell, n_nonstemcell]
    '''
    x, y = xx
    return np.array([
        (2*a-1)*r*x,
        (2*(a+s)-1)*r*y+r*u*x
    ])

def cellnumber_nc(t, x, r, a):
    '''
    Neutral cell number calculator
    
    Args:
        t:
            time
        x:
            neutral cell number
        r, a:
            parameters
    return:
        np.array:
            neutral cell number at time t.
    '''
    N = len(x)
    mat = np.zeros((N, N))
    diag = np.diag_indices_from(mat)
    mat[diag] = -r
    mat[diag[0][1:], diag[1][:-1]] = 2*a*r
    return np.dot(mat, x)

def nc_sol(t, c0, k, r, a):
    '''
    Analytical solution of neutral cell number
    
    Args:
        t:
            time
        c0:
            initial neutral cell number
        k:
            generation
        r, a:
            parameters
    return:
        float:
            neutral cell number in generation k at time t.
    '''
    if k == 0:
        return c0*np.exp(-r*t)
    return c0*(2*np.e*a*r*t/k)**k*np.exp(-r*t)/np.sqrt(2*np.pi*k)

def cellnumber_ac(t, x, c0, r, a, s, u):
    '''
    Neutral cell number calculator
    
    Args:
        t:
            time
        x:
            advantageous cell number
        c0:
            initial ac number
        r, a, s, u:
            parameters
    return:
        np.array:
            advantageous cell number at time t.
    '''
    N = len(x)
    mat, mat1 = np.zeros((N, N)), np.zeros((N, N))
    diag = np.diag_indices_from(mat)
    mat[diag] = -r
    mat[diag[0][1:], diag[1][:-1]] = 2*(a+s)*r
    mat[0][0] = 0
    mat[1][0] = 0
    mat1[diag[0][1:], diag[1][:-1]] = u*r
    nc = np.array([nc_sol(t, c0, k, r, a) for k in range(N)])
    return np.dot(mat, x) + np.dot(mat1, nc)

def p_xi(gen, T, x0, r, a, s, u):
    '''
    Probability density function of LR distance
    Args:
        gen:
            generation
        T: 
            time
        x0:
            initial cell number
        r, a, s, u: 
            parameters
    Return:
        np.array:
            Probability density of LR distance at time T.
    '''
    sol_nc = np.array([nc_sol(T, x0, i, r, a) for i in range(gen)])
    sol_ac = solve_ivp(cellnumber_ac, t_span=(0, T), y0=[0]*gen, method='RK45', args=(x0, r, a, s, u)).y[:,-1]
    cell_num = sol_nc+sol_ac
    return cell_num

def my_loglike(theta, data, args):
    '''
    Likelihood of lr-dist

    Args:
        theta:
            parameters, (r, a, s, u)
        data:
            Observed lr dist
        args:
            paramteres, (time, initial_cell_number, prior_sigma)
    Return:
        float:
            Sum of log-likelihood of given lr dist parameters
    '''
    T, c0, sigma = args
    r, a, s, u = theta
    xt = p_xi(len(data), T, c0, r, a, s, u)
    llh = norm(xt, sigma).logpdf(data).sum()
    return llh

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
            r, a, s, u, c0 = theta
            u = 10**-(u*5)
            return -my_loglike((r, a, s, u), data, (T, c0, sigma))
    else:
        def loss(theta):
            r, a, s, u = theta
            u = 10**-(u*5)
            return -my_loglike((r, a, s, u), data, (T, c0, sigma)) 
    
    def run1():
        if c0 is None:
            de = DE(func=loss, n_dim=5, max_iter=100, lb=[0, 0.5, 0, 1, 1], ub=[5, 1, 0.5, 10, 2000])
        else:
            de = DE(func=loss, n_dim=4, max_iter=100, lb=[0, 0.5, 0, 1], ub=[5, 1, 0.5, 10])
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
    

class LogLike(pt.Op):
    itypes = [pt.dvector]
    otypes = [pt.dscalar]

    def __init__(self, loglike, data, args):
        
        # add inputs as class attributes
        self.likelihood = loglike
        self.data = data
        self.args = args

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        (theta,) = inputs  # this will contain my variables
 
        # call the log-likelihood function
        logl = self.likelihood(theta, self.data, self.args)

        outputs[0][0] = np.array(logl)  # output the log-likelihood
        
def mcmc_inference(data, para_prior, T, c0, sigma, draw=1000, tune=1000, chains=5):
    '''
    Mutation rate estimation using DE-MCMC
    
    Args:
        data:
            Observed lp-dist
        data_prior:
            mean of prior distributions of all parameters
        T:
            time of phylodynamics eqns
        c0:
            initial cell numbers
        sigma:
            variation of loss function
        draw:
            Number of smaples to draw
        tune:
            Number of iterations to tune
        chain:
            number of chains to sample
    '''
    logl = LogLike(my_loglike, data, (T, c0, sigma))
    rh, ah, sh, uh = para_prior
    with pm.Model() as model:
        r = pm.TruncatedNormal('r', mu=rh, sigma=0.1, lower=0, upper=5, initval=rh)
        a = pm.TruncatedNormal('a', mu=ah, sigma=0.1, lower=0.5, upper=1, initval=ah)
        s = pm.TruncatedNormal('s', mu=sh, sigma=0.1, lower=0, upper=0.5, initval=sh)
        u = pm.Beta('u', alpha=1, beta=1/uh-1, initval=uh)
        theta = pt.as_tensor_variable([r, a, s, u])
        pm.Potential("likelihood", logl(theta))
        idata = pm.sample(draw, tune=tune, step=pm.DEMetropolis(), chains=chains)
    return idata



