from sympy import polylog
import numpy as np
from sko.DE import DE
from scipy.stats import gaussian_kde, poisson
from scipy.optimize import minimize
import pymc as pm
import pytensor
import pytensor.tensor as pt

class GenerationEst:
    '''
    Estimation of generations with given LR distance distribution and mutaion rate
    
    Args:
        lr_dist: 
            distribution of LR distance
        mu: 
            mutation rate
    '''
    def __init__(self, lr_dist, mu:float, gennum=None):
        self.lr_dist = lr_dist
        self.mu = mu
        # self.pm = gaussian_kde(mutnum)
        # self.pmg = lambda g: poisson(g*mu)
        self.u1 = np.mean(lr_dist)/mu
        self.s1 = np.std(lr_dist)/mu
        self.generation = None
        if gennum is None:
            self.pg = gaussian_kde(np.array(lr_dist)/mu)
        else:
            self.pg = gaussian_kde(gennum)
        
    def generation_map(self, mn:float):
        '''
        MAP estimation of given lr-dist
        
        Args:
            mn: 
                lr-distance
            
        Returns:
            float:
                Estimated generation
        '''
        # prob = lambda g: -self.pmg(g).pmf(mn)*self.pg.pdf(g)/self.pm.pdf(mn)
        # res = minimize(prob, mn, bounds=[(0,2*mn/self.mu)])
        # return res.x[0]
        u1, s1 = self.u1, self.s1
        return 0.5*(u1-s1**2*self.mu+np.sqrt(4*mn*s1**2+(-u1+s1**2*self.mu)**2))
    
    def estimate(self, cell_number=None):
        '''
        Generation estimation of all given lr distance in self.lr_dist
        
        '''
        lr_unique = list(set(self.lr_dist))
        mg_map = dict()
        for i in lr_unique:
            mg_map[i] = self.generation_map(i)
        generation = np.array([mg_map[i] for i in self.lr_dist])
        if cell_number is None:
            cell_number = len(generation)
        gen_kde = gaussian_kde(generation)
        max_gen = 1+int(max(generation))
        gen_num = np.array([gen_kde.pdf(i) for i in range(max_gen+1)])
        gen_num = gen_num / gen_num.sum() * cell_number
        return gen_num.flatten()
    
class BranchLength:
    '''
    Probability distribution of LP distance
    
    Args:
        mu:
            mutation rate
        delta:
            Probability of branching division
    '''
    def __init__(self, mu, delta):
        self.mu = mu
        self.delta = delta
    
    def prob(self, x):
        '''
        Probability density function of lp-dist
        
        Args:
            x:
                lp-dist
        Return:
            float:
                probability density
        '''
        mu, delta = self.mu, self.delta
        x = float(x)
        if x == 0:
            coef = (mu)**x*delta/(1-delta)
        else:
            coef = (mu)**x*delta/(1-delta)/(np.sqrt(2*np.pi*x)*(x/np.e)**x)
        return coef*float(polylog(-x, np.exp(-mu)*(1-delta)))
    
    def likelihood(self, data):
        '''
        Likelihood of lp-dist
        
        Args:
            data:
                Observed lp dist
        Return:
            float:
                Sum of log-likelihood of given lp dist
        '''
        unique_data = list(set(data))
        pre_comp = np.array([self.prob(i) for i in range(int(max(unique_data))+1)])
        pre_comp = np.log(pre_comp / np.sum(pre_comp)+1e-99)
        lh = 0
        for i in data:
            lh += pre_comp[int(i)]
        return lh
    
def mutation_rate_de(data, n_iter:int=100, bootstrape:int=0):
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
    def loss(theta):
        '''
        Optimization loss function
        
        Args:
            theta:
                lp dist parameter, (mu, delta)
        Return:
            float:
                - log-liklihood function of given parameters
        '''
        mu, delta = theta
        bl = BranchLength(mu, delta)
        return -bl.likelihood(data)
    
    def run1():
        '''
        run DE-estimation
        

        '''
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
    mu, delta = theta
    return BranchLength(mu, delta).likelihood(data)

def mutation_rate_mcmc(data, draw=1000, tune=1000, chain=4, mu0=2, sigma=0.2):
    '''
    Mutation rate estimation using DE-MCMC
    
    Args:
        data:
            Observed lp-dist
        draw:
            Number of smaples to draw
        tune:
            Number of iterations to tune
        chain:
            number of chains to sample
        mu0:
            mean of prior distribution of mutation rate
        sigma:
            variation of prior distribution of mutation rate
    '''
    logl = LogLike(my_loglike, data)
    # muh, betah = para_prior
    with pm.Model() as model:
        mu = pm.TruncatedNormal('mu', mu=mu0, sigma=sigma, lower=0, upper=10)
        delta = pm.Beta('delta', alpha=1, beta=1)
        theta = pt.as_tensor_variable([mu, delta])
        pm.Potential("likelihood", logl(theta))
        idata = pm.sample(draw, tune=tune, step=pm.DEMetropolis(), chains=4)
    return idata
    
    