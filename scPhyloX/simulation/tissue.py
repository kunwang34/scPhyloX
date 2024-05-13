import sys
import numpy as np
from scipy.special import comb, factorial
from copy import deepcopy
from tqdm import tqdm
from collections import defaultdict
import pandas as pd

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

class Reaction: 
    '''
    Cell division/differentiation type
    
    Args:
        rate:
            reaction rate function
        num_lefts:
            Cell numbers before reaction
        num_right:
            Cell numbers after reaction
        index:
            Reaction index
    '''
    def __init__(self, rate=0., num_lefts=None, num_rights=None, index=None):
        self.rate = rate
        assert len(num_lefts) == len(num_rights)
        self.num_lefts = np.array(num_lefts)
        self.num_rights = np.array(num_rights)
        self.num_diff = self.num_rights - self.num_lefts
        self.index = index
    def combine(self, n, s):
        return np.prod(comb(n, s))
    def propensity(self, n, t): 
        return self.rate(t) * self.combine(n, self.num_lefts)

class Cell:
    """
    Cell class
    
    Args:
        seq:
            DNA sequence
        gen: 
            Cell generation
        cellid: 
            Cell id
        celltype:
            Cell type, stem/non-stem cell
        is_alive: 
            Is cell alive or died
        lseq: 
            Length of DNA seq 
        init: 
            if init, cell will generate DNA seq with given length lseq automatically
    """
    def __init__(self, seq=None, gen=None, cellid=None, celltype=None, is_alive=True, lseq=1500, init=False):
        self.seq = seq
        self.gen = gen
        self.celltype= celltype
        self.lseq = lseq
        self.cellid=cellid
        self.is_alive = is_alive
        if init:
            self.initialize(lseq)  
    def initialize(self, lseq):
        '''
        Generate DNA seq with given length lseq
        '''
        if self.seq is None:
            self.seq = np.zeros(lseq)
        if self.gen is None:
            self.gen = 0
            
class System: 
    '''
    Gillespie simulation
    
    Args:
        num_elements: 
            Cell type number
        inits: 
            Initial cell number
        nbase:
            length of cell DNA seq
        mut_rate:
            mutation rate of DNA seq, follows Poisson distribution
        max_t:
            maximum simulation time
        start_t:
            Mutate start time
    '''
    def __init__(self, num_elements, inits=None, nbase=1500,
                 mut_rate=1, max_t=35, start_t=0):
        assert num_elements > 0
        self.num_elements = num_elements
        self.reactions = []
        self.start_t = start_t
        if inits is None:
            self.n = [np.ones(self.num_elements)]
        else:
            self.n = [np.array(inits)]
        self.max_t = max_t
        self.mut_rate = mut_rate
        self.global_id = defaultdict(int)
        self.Stemcells = [Cell(init=True, celltype='c', lseq=nbase, cellid=_) for _ in range(int(self.n[0][0]))]
        self.Diffcells = [Cell(init=True, celltype='n', lseq=nbase, cellid=_) for _ in range(int(self.n[0][1]))]
        self.global_id[0] += np.sum(self.n[0])
        self.mut_num = 0
        self.cell_num = sum(self.n[0])   
        self.mut_time = []
        
        self.mut_num_SC = [0]*int(self.n[0][0])
        self.mut_num_DC = [0]*int(self.n[0][1])
        self.log_cells = dict()
        self.lineage_info = dict()
        for cell in self.Stemcells:
            self.lineage_info[f'<{cell.gen}_{cell.cellid}>'] = 0
            self.log_cells[f'<{cell.gen}_{cell.cellid}>'] = cell
        for cell in self.Diffcells:
            self.lineage_info[f'<{cell.gen}_{cell.cellid}>'] = 0
            self.log_cells[f'<{cell.gen}_{cell.cellid}>'] = cell
            
    def add_reaction(self, rate=0., num_lefts=None, num_rights=None,index=None):
        '''
        Add reactions to simulation
        
        Args:
            rate:
                reaction rate function
            num_lefts:
                Cell numbers before reaction
            num_right:
                Cell numbers after reaction
            index:
                Reaction index
        '''
        assert len(num_lefts) == self.num_elements
        assert len(num_rights) == self.num_elements
        self.reactions.append(Reaction(rate, num_lefts, num_rights,index))
    
    def mutate(self, cell, mutrate):
        '''
        simulation DNA mutation
        
        Args:
            cell:
                cell
            mutrate:
                mutation rate
        Return:
            cell:
                cell with mutated DNA seq
        '''
        if cell.gen < self.start_t:
            return cell
        mut_num = np.random.poisson(mutrate)
        cell.seq[np.random.choice(range(cell.lseq), mut_num)] = 1
        return cell
    
    def stemrenewal(self):
        '''
        stem cell -> 2 stem cells
        '''
        ind = np.random.choice(range(len(self.Stemcells)))
        des1 = deepcopy(self.Stemcells[ind])
        des2 = deepcopy(self.Stemcells[ind])
        self.log_cells[f'<{self.Stemcells[ind].gen}_{self.Stemcells[ind].cellid}>'].is_alive=False
        des1 = self.mutate(des1, self.mut_rate)
        des2 = self.mutate(des2, self.mut_rate)
        des1.gen += 1
        des2.gen += 1
        des1.cellid = self.global_id[des1.gen]
        des2.cellid = self.global_id[des2.gen] + 1
        self.global_id[des1.gen] += 2
        self.lineage_info[f'<{des1.gen}_{des1.cellid}>'] = self.Stemcells[ind].cellid
        self.lineage_info[f'<{des2.gen}_{des2.cellid}>'] = self.Stemcells[ind].cellid
        self.log_cells[f'<{des1.gen}_{des1.cellid}>'] = des1
        self.log_cells[f'<{des2.gen}_{des2.cellid}>'] = des2
        self.mut_num -= self.mut_num_SC[ind]
        del self.Stemcells[ind]
        del self.mut_num_SC[ind]
        
        self.Stemcells.append(des1)
        self.Stemcells.append(des2)
        self.mut_num_SC.append(sum(self.Stemcells[-1].seq))
        self.mut_num_SC.append(sum(self.Stemcells[-2].seq))
        self.mut_num += sum(self.mut_num_SC[-2:])
        self.cell_num += 1
        
    def stemdiff(self):
        '''
        stem cell -> non-stem cell + non-stem cell
        '''
        ind = np.random.choice(range(len(self.Stemcells)))
        des1 = deepcopy(self.Stemcells[ind])
        des2 = deepcopy(self.Stemcells[ind])
        self.log_cells[f'<{self.Stemcells[ind].gen}_{self.Stemcells[ind].cellid}>'].is_alive=False
        self.mut_num -= self.mut_num_SC[ind]
        des1.celltype = 'n'
        des2.celltype = 'n'
        des1.gen += 1
        des2.gen += 1
        des1.cellid = self.global_id[des1.gen]
        des2.cellid = self.global_id[des2.gen] + 1
        self.lineage_info[f'<{des1.gen}_{des1.cellid}>'] = self.Stemcells[ind].cellid
        self.lineage_info[f'<{des2.gen}_{des2.cellid}>'] = self.Stemcells[ind].cellid
        self.log_cells[f'<{des1.gen}_{des1.cellid}>'] = des1
        self.log_cells[f'<{des2.gen}_{des2.cellid}>'] = des2
        self.global_id[des1.gen] += 2
        del self.Stemcells[ind]
        del self.mut_num_SC[ind]
        self.Diffcells.append(self.mutate(des1, self.mut_rate))
        self.Diffcells.append(self.mutate(des2, self.mut_rate))
        self.mut_num_DC.append(sum(self.Diffcells[-1].seq))
        self.mut_num_DC.append(sum(self.Diffcells[-2].seq)) 
        self.mut_num += sum(self.mut_num_DC[-2:])
        self.cell_num += 1
        
    def diffdeath(self):
        '''
        non-stem cell -> death cell
        '''
        ind = np.random.choice(range(len(self.Diffcells)))
        self.mut_num -= self.mut_num_DC[ind]
        self.cell_num -= 1
        self.mut_num -= sum(self.Diffcells[ind].seq)
        self.log_cells[f'<{self.Diffcells[ind].gen}_{self.Diffcells[ind].cellid}>'].is_alive=False
        del self.Diffcells[ind]   
        del self.mut_num_DC[ind]
        
    def stemasym(self):
        '''
        stem cell -> stem cell + non-stem cell
        '''
        ind = np.random.choice(range(len(self.Stemcells)))
        des1 = deepcopy(self.Stemcells[ind])
        des2 = deepcopy(self.Stemcells[ind])
        self.log_cells[f'<{self.Stemcells[ind].gen}_{self.Stemcells[ind].cellid}>'].is_alive=False
        self.mut_num -= self.mut_num_SC[ind]
        des2.celltype = 'n'
        des1.gen += 1
        des2.gen += 1
        des1.cellid = self.global_id[des1.gen]
        des2.cellid = self.global_id[des2.gen] + 1
        self.lineage_info[f'<{des1.gen}_{des1.cellid}>'] = self.Stemcells[ind].cellid
        self.lineage_info[f'<{des2.gen}_{des2.cellid}>'] = self.Stemcells[ind].cellid
        self.log_cells[f'<{des1.gen}_{des1.cellid}>'] = des1
        self.log_cells[f'<{des2.gen}_{des2.cellid}>'] = des2
        self.global_id[des1.gen] += 2
        self.Stemcells.append(self.mutate(des1, self.mut_rate))
        self.Diffcells.append(self.mutate(des2, self.mut_rate))
        self.mut_num_SC.append(sum(self.Stemcells[-1].seq))
        self.mut_num_DC.append(sum(self.Diffcells[-1].seq)) 
        self.mut_num += self.mut_num_SC[-1]
        self.mut_num += self.mut_num_DC[-1]
        self.cell_num += 1
        
        del self.Stemcells[ind]
        del self.mut_num_SC[ind]

    def evolute(self, steps): 
        self.t = [0]
        self.log = []
        for i in range(steps):
            assert len(self.t) == len(self.n)
            A = np.array([rec.propensity(self.n[-1], self.t[-1])
                          for rec in self.reactions]) 
            A0 = A.sum() 
            A /= A0   
            t0 = -np.log(np.random.random())/A0
            self.t.append(self.t[-1] + t0)
            d = np.random.choice(self.reactions, p=A)
            self.n.append(self.n[-1] + d.num_diff)
            switch = {1:self.stemrenewal,2:self.stemdiff,3:self.diffdeath, 4:self.stemasym}
            switch.get(d.index)()
            self.log.append(d.index) 
            self.mut_time.append(self.mut_num/self.cell_num)
            print(f'\rcell_num:{self.cell_num}, time:{self.t[-1]}',end = "")
            if self.t[-1] > self.max_t:
                break
            
def simulation(x0, max_t, mut_rate, a, b, p, r, k, d, t0):
    '''
    Run gillespie simulation in tissue development model
    
    Args:
        x0:
            initial cell number
        max_t:
            stop time
        mut_rate:
            mutation rate
        a,b,p,r,k,d,t0:
            paras
    Return:
        Object:
            gillespie simulator with results
    '''
    num_elements = 2
    system = System(num_elements,inits = x0, max_t=max_t, mut_rate=mut_rate)

    beta = lambda t: bt(t, a, b, k, t0)
    
    birth = lambda t: r*beta(t)
    div_asym = lambda t: r*(1-beta(t))*(1-p)
    div_sym = lambda t: r*(1-beta(t))*p
    death = lambda t: d
    
    system.add_reaction(birth, [1, 0], [2, 0], 1)
    system.add_reaction(div_sym, [1, 0], [0, 2], 2)
    system.add_reaction(death, [0, 1], [0, 0], 3)
    system.add_reaction(div_asym, [1, 0], [1, 1], 4)
    system.evolute(2000000000)
    
    return system
