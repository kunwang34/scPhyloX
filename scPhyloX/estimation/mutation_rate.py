from sympy import polylog
import numpy as np
from sko.DE import DE
from scipy.stats import gaussian_kde, poisson
from scipy.optimize import minimize
import pymc as pm
import pytensor
import pytensor.tensor as pt

class GenerationEst:
    def __init__(self, mutnum, mu, gennum=None):
        self.mutnum = mutnum
        self.mu = mu
        # self.pm = gaussian_kde(mutnum)
        # self.pmg = lambda g: poisson(g*mu)
        self.u1 = np.mean(mutnum)/mu
        self.s1 = np.std(mutnum)/mu
        self.generation = None
        if gennum is None:
            self.pg = gaussian_kde(np.array(mutnum)/mu)
        else:
            self.pg = gaussian_kde(gennum)
        
    def generation_map(self, mn):
        # prob = lambda g: -self.pmg(g).pmf(mn)*self.pg.pdf(g)/self.pm.pdf(mn)
        # res = minimize(prob, mn, bounds=[(0,2*mn/self.mu)])
        # return res.x[0]
        u1, s1 = self.u1, self.s1
        return 0.5*(u1-s1**2*self.mu+np.sqrt(4*mn*s1**2+(-u1+s1**2*self.mu)**2))
    
    def estimate(self, cell_number=None):
        mutnum_unique = list(set(self.mutnum))
        mg_map = dict()
        for i in mutnum_unique:
            mg_map[i] = self.generation_map(i)
        generation = np.array([mg_map[i] for i in self.mutnum])
        if cell_number is None:
            cell_number = len(generation)
        gen_kde = gaussian_kde(generation)
        max_gen = 1+int(max(generation))
        gen_num = np.array([gen_kde.pdf(i) for i in range(max_gen+1)])
        gen_num = gen_num / gen_num.sum() * cell_number
        return gen_num.flatten()
    
class BranchLength:
    def __init__(self, mu, beta):
        self.mu = mu
        self.beta = beta
    
    def prob(self, x):
        mu, beta = self.mu, self.beta
        x = float(x)
        if x == 0:
            coef = (mu)**x*beta/(1-beta)
        else:
            coef = (mu)**x*beta/(1-beta)/(np.sqrt(2*np.pi*x)*(x/np.e)**x)
        return coef*float(polylog(-x, np.exp(-mu)*(1-beta)))
    
    def likelihood(self, data):
        unique_data = list(set(data))
        pre_comp = np.array([self.prob(i) for i in range(int(max(unique_data))+1)])
        pre_comp = np.log(pre_comp / np.sum(pre_comp)+1e-99)
        lh = 0
        for i in data:
            lh += pre_comp[int(i)]
        return lh
    
def mutation_rate_de(data, n_iter=100, bootstrape=0):
    def loss(theta):
        mu, beta = theta
        bl = BranchLength(mu, beta)
        return -bl.likelihood(data)
    
    def run1():
        de = DE(func=loss, n_dim=2, max_iter=n_iter, lb=[0, 0], ub=[10, 1])
        bestx, besty = [], []
        for i in range(n_iter):
            xt,yt = de.run(1)
            bestx.append(xt)
            besty.append(yt)
            print(f'\riter{i}, loss:{yt}, est={xt}',end = "")
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

    def __init__(self, loglike, data):
        
        # add inputs as class attributes
        self.likelihood = loglike
        self.data = data

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        (theta,) = inputs  # this will contain my variables
        # call the log-likelihood function
        logl = self.likelihood(theta, self.data)
        outputs[0][0] = np.array(logl)  # output the log-likelihood
        
def my_loglike(theta, data):
    mu, beta = theta
    return BranchLength(mu, beta).likelihood(data)

def mutation_rate_mcmc(data, draw=1000, tune=1000, chain=4, mu0=2, sigma=0.2):
    logl = LogLike(my_loglike, data)
    # muh, betah = para_prior
    with pm.Model() as model:
        mu = pm.TruncatedNormal('mu', mu=mu0, sigma=sigma, lower=0, upper=10)
        delta = pm.Beta('delta', alpha=1, beta=1)
        theta = pt.as_tensor_variable([mu, delta])
        pm.Potential("likelihood", logl(theta))
        idata = pm.sample(draw, tune=tune, step=pm.DEMetropolis(), chains=4)
    return idata
    
    