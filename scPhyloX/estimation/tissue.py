import numpy as np
from scipy.special import factorial
from sympy import polylog
from scipy.stats import norm, beta
import pymc as pm
import pytensor
import pytensor.tensor as pt
import arviz as az
from scipy.integrate import quad
from sko.DE import DE
import warnings

def bt(t, a, b, k, t0):
    # a+b -> b
    return a/(1+np.exp(k*(t-t0))) + b


def cellnumber(t, xx, a, b, k, t0, p, r, d):
    x, y = xx
    bt = lambda t: a/(1+np.exp(k*(t-t0))) + b
    return np.array([
        ((1+p)*bt(t)-p)*r*x,
        (1-bt(t))*(1+p)*r*x -d*y
    ])
    
def ncyc(i, t, c0, ax, bx, r, d, k, t0):
    if i == 0 :
        t1 = c0*np.exp(-r*t)
    else:
        t1 = c0*np.exp(-r*t)*(r*np.e/(k*i))**i/np.sqrt(2*np.pi*i)
    if not np.isfinite(t1):
        print(ax, bx, r, d, k, t0)
    t2 = ((ax+bx)*k*t+ax*np.log((1+np.exp(-k*t0))/(1+np.exp(k*(t-t0)))))**i
    return t1*t2

def nnc(i, t, c0, ax, bx, r, d, k, t0):
    func = lambda t1: np.exp(d*t1)*(2-ax/(1+np.exp(k*(t1-t0)))-bx)*ncyc(i, t1, c0, ax, bx, r, d, k, t0)
    res = np.exp(-d*t)*r*quad(func, 0, t)[0]
    return res

def p_xi(n_gen, T, c0, ax, bx, r, d, k, t0):
    xt = np.array([ncyc(i, T, c0, ax, bx, r, d, k, t0) for i in range(n_gen)])+ np.array([nnc(i, T, c0, ax, bx, r, d, k, t0) for i in range(n_gen)])
    return xt

def my_loglike(theta, data, args):
    T, c0, sigma = args
    ax, bx, r, d, k, t0 = theta
    xt = p_xi(len(data), T, c0, ax, bx, r, d, k, t0)
    llh = norm(xt, sigma).logpdf(data).sum()
    return llh

def para_inference_DE(data, T=20, c0=None, sigma=1, n_iter=100, bootstrape=0, verbose='text'):
    if c0 is None:
        def loss(theta):
            ax, bx, r, d, k, t0, c0 = theta
            d = 10**(-d)
            return -my_loglike((ax, bx, r, d, k, t0), data, (T, c0, sigma))
    else:
        def loss(theta):
            ax, bx, r, d, k, t0 = theta
            d = 10**(-d)
            return -my_loglike((ax, bx, r, d, k, t0), data, (T, c0, sigma)) 
    
    def run1():
        constraint_ueq = [lambda x: x[0]+x[1]-2]
        if c0 is None:
            de = DE(func=loss, n_dim=7, max_iter=n_iter, lb=[0, 0, 0, 1, 0, 0, 1], ub=[2, 2, 3, 5, 3, 20, 2000], constraint_ueq=constraint_ueq)
        else:
            de = DE(func=loss, n_dim=6, max_iter=n_iter, lb=[0, 0, 0, 1, 0, 0], ub=[2, 2, 3, 5, 3, 20], constraint_ueq=constraint_ueq)
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
        
def mcmc_inference(data, para_prior, T, c0, sigma, draw=1000, tune=1000, chains=8):
    logl = LogLike(my_loglike, data, (T, c0, sigma))
    axh, bxh, rh, dh, kh, t0h = para_prior
    with pm.Model() as model:
        ax = pm.TruncatedNormal('ax', mu=axh, sigma=0.1, lower=0, upper=2, initval=axh)
        bx = pm.TruncatedNormal('bx', mu=bxh, sigma=0.1, lower=0, upper=2, initval=bxh)
        r = pm.TruncatedNormal('r', mu=rh, sigma=0.1, lower=0.1, initval=rh)
        k = pm.TruncatedNormal('k', mu=kh, sigma=0.1, lower=0, upper=3, initval=kh)
        t0 = pm.TruncatedNormal('t0', mu=t0h, sigma=0.3, lower=0, upper=20, initval=t0h)
        d = pm.Beta('d', alpha=1, beta=1/dh-1, initval=dh)
        theta = pt.as_tensor_variable([ax, bx, r, d, k, t0])
        pm.Potential("likelihood", logl(theta))
        idata = pm.sample(draw, tune=tune, step=pm.DEMetropolis(), chains=chains)
    return idata
