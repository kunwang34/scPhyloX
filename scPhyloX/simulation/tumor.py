import sys
import numpy as np
from scipy.special import comb, factorial
from copy import deepcopy
from tqdm import tqdm
from collections import defaultdict
import pandas as pd

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
            Cell type, neutral/advantageous cell
        is_alive: 
            Is cell alive or died
        lseq: 
            Length of DNA seq 
        init: 
            if init, cell will generate DNA seq with given length lseq automatically
    """ 
    def __init__(self, seq=None, gen=None, cellid=None, celltype=None, is_alive=True, lseq=10000, init=False):
        self.seq = seq
        self.gen = gen
        self.celltype= celltype
        self.lseq = lseq
        self.cellid=cellid
        self.is_alive = is_alive
        if init:
            self.initialize(lseq)  
    def initialize(self, lseq):
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
        self.Neutral = [Cell(init=True, celltype='c', lseq=nbase, cellid=_) for _ in range(int(self.n[0][0]))]
        self.Advantageous = [Cell(init=True, celltype='n', lseq=nbase, cellid=_) for _ in range(int(self.n[0][1]))]
        self.global_id[0] += np.sum(self.n[0])
        self.mut_num = 0
        self.cell_num = sum(self.n[0])   
        self.mut_time = []
        
        self.mut_num_NC = [0]*int(self.n[0][0])
        self.mut_num_AC = [0]*int(self.n[0][1])
        self.log_cells = dict()
        self.lineage_info = dict()
        for cell in self.Neutral:
            self.lineage_info[f'<{cell.gen}_{cell.cellid}>'] = 0
            self.log_cells[f'<{cell.gen}_{cell.cellid}>'] = cell
        for cell in self.Advantageous:
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
    
    def ncrenewal(self):
        '''
        neutral cell -> 2 neutral cells
        '''
        ind = np.random.choice(range(len(self.Neutral)))
        des1 = deepcopy(self.Neutral[ind])
        des2 = deepcopy(self.Neutral[ind])
        self.log_cells[f'<{self.Neutral[ind].gen}_{self.Neutral[ind].cellid}>'].is_alive=False
        des1 = self.mutate(des1, self.mut_rate)
        des2 = self.mutate(des2, self.mut_rate)
        des1.gen += 1
        des2.gen += 1
        des1.cellid = self.global_id[des1.gen]
        des2.cellid = self.global_id[des2.gen] + 1
        self.global_id[des1.gen] += 2
        self.lineage_info[f'<{des1.gen}_{des1.cellid}>'] = self.Neutral[ind].cellid
        self.lineage_info[f'<{des2.gen}_{des2.cellid}>'] = self.Neutral[ind].cellid
        self.log_cells[f'<{des1.gen}_{des1.cellid}>'] = des1
        self.log_cells[f'<{des2.gen}_{des2.cellid}>'] = des2
        self.mut_num -= self.mut_num_NC[ind]
        del self.Neutral[ind]
        del self.mut_num_NC[ind]
        
        self.Neutral.append(des1)
        self.Neutral.append(des2)
        self.mut_num_NC.append(sum(self.Neutral[-1].seq))
        self.mut_num_NC.append(sum(self.Neutral[-2].seq))
        self.mut_num += sum(self.mut_num_NC[-2:])
        self.cell_num += 1
    
    def acrenewal(self):
        '''
        advantageous cell -> 2 advantageous cells
        '''
        ind = np.random.choice(range(len(self.Advantageous)))
        des1 = deepcopy(self.Advantageous[ind])
        des2 = deepcopy(self.Advantageous[ind])
        self.log_cells[f'<{self.Advantageous[ind].gen}_{self.Advantageous[ind].cellid}>'].is_alive=False
        des1 = self.mutate(des1, self.mut_rate)
        des2 = self.mutate(des2, self.mut_rate)
        des1.gen += 1
        des2.gen += 1
        des1.cellid = self.global_id[des1.gen]
        des2.cellid = self.global_id[des2.gen] + 1
        self.global_id[des1.gen] += 2
        self.lineage_info[f'<{des1.gen}_{des1.cellid}>'] = self.Advantageous[ind].cellid
        self.lineage_info[f'<{des2.gen}_{des2.cellid}>'] = self.Advantageous[ind].cellid
        self.log_cells[f'<{des1.gen}_{des1.cellid}>'] = des1
        self.log_cells[f'<{des2.gen}_{des2.cellid}>'] = des2
        self.mut_num -= self.mut_num_AC[ind]
        del self.Advantageous[ind]
        del self.mut_num_AC[ind]
        
        self.Advantageous.append(des1)
        self.Advantageous.append(des2)
        self.mut_num_AC.append(sum(self.Advantageous[-1].seq))
        self.mut_num_AC.append(sum(self.Advantageous[-2].seq))
        self.mut_num += sum(self.mut_num_AC[-2:])
        self.cell_num += 1
        
    def advmut(self):
        '''
        neutral cell -> advantageous cell
        '''
        ind = np.random.choice(range(len(self.Neutral)))
        des1 = deepcopy(self.Neutral[ind])
        self.log_cells[f'<{self.Neutral[ind].gen}_{self.Neutral[ind].cellid}>'].is_alive=False
        self.mut_num -= self.mut_num_NC[ind]
        des1.celltype = 'n'
        des1.gen += 1
        des1.cellid = self.global_id[des1.gen]
        self.lineage_info[f'<{des1.gen}_{des1.cellid}>'] = self.Neutral[ind].cellid
        self.log_cells[f'<{des1.gen}_{des1.cellid}>'] = des1
        self.global_id[des1.gen] += 1
        del self.Neutral[ind]
        del self.mut_num_NC[ind]
        self.Advantageous.append(self.mutate(des1, self.mut_rate))
        self.mut_num_AC.append(sum(self.Advantageous[-1].seq))
        self.mut_num += self.mut_num_AC[-1]
        self.cell_num += 1
    
    def ncdeath(self):
        '''
        neutral cell -> death
        '''
        ind = np.random.choice(range(len(self.Neutral)))
        self.mut_num -= self.mut_num_NC[ind]
        self.cell_num -= 1
        self.mut_num -= sum(self.Neutral[ind].seq)
        self.log_cells[f'<{self.Neutral[ind].gen}_{self.Neutral[ind].cellid}>'].is_alive=False
        del self.Neutral[ind]   
        del self.mut_num_NC[ind]
        
    def acdeath(self):
        '''
        advantageous cell -> death
        '''
        ind = np.random.choice(range(len(self.Advantageous)))
        self.mut_num -= self.mut_num_AC[ind]
        self.cell_num -= 1
        self.mut_num -= sum(self.Advantageous[ind].seq)
        self.log_cells[f'<{self.Advantageous[ind].gen}_{self.Advantageous[ind].cellid}>'].is_alive=False
        del self.Advantageous[ind]   
        del self.mut_num_AC[ind]
        

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
            switch = {1:self.ncrenewal,2:self.acrenewal,3:self.advmut, 4:self.ncdeath, 5:self.acdeath}
            switch.get(d.index)()
            self.log.append(d.index) 
            self.mut_time.append(self.mut_num/self.cell_num)
            print(f'\rcell_num:{self.cell_num}, time:{self.t[-1]}',end = "")
            if self.t[-1] > self.max_t:
                break

def simulation(x0, max_t, mut_rate, r, a, s, u):
    num_elements = 2
    system = System(num_elements, inits = x0, max_t = max_t, mut_rate = mut_rate)

    birth_n = lambda t: r*a
    birth_a = lambda t: r*(a+s)
    diff = lambda t: r*u
    death_n = lambda t: r*(1-a-u)
    death_a = lambda t: r*(1-a-s)

    system.add_reaction(birth_n, [1, 0], [2, 0], 1)
    system.add_reaction(birth_a, [0, 1], [0, 2], 2)
    system.add_reaction(diff, [1, 0], [0, 1], 3)
    system.add_reaction(death_n, [1, 0], [0, 0], 4)
    system.add_reaction(death_a, [0, 1], [0, 0], 5)
    system.evolute(200000000)
    return system 