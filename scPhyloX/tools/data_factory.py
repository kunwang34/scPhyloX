import numpy as np
from tqdm import tqdm
from Bio import Phylo

population_size = {'Br':75000,'Ey':44000,'Fb':281600,'L1':10000,'L2':13500,'L3':17000,'Mp':19470,'Sg':102400,'Wg':49000}
# population_size = {'Br':75000,'Ey':44000,'Fb':2200,'L1':10000,'L2':13500,'L3':17000,'Mp':19470,'Sg':200,'Wg':49000}
def cell_number_calc(w, l, r, coef=1e8):
    vol = 4*np.pi*w*l*min(w,l)/6
    vol = vol/r**3
    return vol*coef
tumor_data = {'4_T':(20,55,91), '5_T':(14,32,91), '16_T':(18,34,81), '19_T3':(18,32,75), '49_T1':(35,46,120),  '49_T3':(31,64,120), '65_T1':(31,47,115), '66_T':(36,53,117)}
tumor_size = dict()
for i in tumor_data:
    tumor_size[i] = cell_number_calc(*tumor_data[i])


def get_branchlen(seqtab:np.ndarray=None, sys=None):
    if seqtab is None:
        assert not sys is None
        seqtab = np.array([i.seq for i in sys.Stemcells] + [i.seq for i in sys.Diffcells])

    ncells = seqtab.shape[0]
    distmat = np.zeros((ncells, ncells))
    with tqdm(total=int(ncells*(ncells+1)/2)) as pbar:
        for i in range(ncells):
            for j in range(i, ncells): 
                pbar.update(1)
                if i == j:
                    d1, d2 = 99999, 99999
                else:
                    anc = np.min(seqtab[[i, j],:], 0)
                    d1 = np.sum(np.logical_xor(seqtab[i], anc))
                    d2 = np.sum(np.logical_xor(seqtab[j], anc))
                distmat[i,j] = d1
                distmat[j,i] = d2
    return np.min(distmat, 1)

def get_mutnum(seqtab=None, sys=None, filter_trunk=True):
    if seqtab is None:
        assert not sys is None
        seqtab = np.array([i.seq for i in sys.Stemcells] + [i.seq for i in sys.Diffcells])
    # if filter_trunk:
    #     seqtab
    return np.sum(seqtab, axis=1)

# def mutnum_fly(seqs, ref):
#     with open(ref, 'r') as f:
#         ref0 = f.readlines()[-1]
#     ref = [i for i in ref0 if i in 'CG']
#     with open(seqs, 'r') as f:
#         seq = f.readlines()
#     res = []
#     for i in seq:
#         tmp = 0
#         for j in range(len(i)-1):
#             tmp += i[j]!=ref[j]
#         res.append(tmp)
#     return np.array(res)

def mutnum_fly(seqs, ref):
    with open(ref, 'r') as f:
        ref0 = f.readlines()[-1]
    ref = [i for i in ref0 if i in 'CG']
    with open(seqs, 'r') as f:
        seq = f.readlines()
    mutmat = []
    for i in seq:
        mutarr = []
        for j in range(len(i)-1):
            mutarr.append(i[j]!=ref[j])
        mutmat.append(mutarr)
    mutmat = np.unique(np.array(mutmat), axis=0).astype(int)
    return np.sum(mutmat, axis=1)


def branchlen_fly(seqs, ref, rs=1):
    with open(ref, 'r') as f:
        ref0 = f.readlines()[-1]
    ref = [i for i in ref0 if i in 'CG']
    with open(seqs, 'r') as f:
        seq = f.readlines()
    mutmat = []
    for i in seq:
        mutarr = []
        for j in range(len(i)-1):
            mutarr.append(i[j]!=ref[j])
        mutmat.append(mutarr)
    mutmat = np.unique(np.array(mutmat), axis=0).astype(int)
    if rs != 1:
        mutmat = mutmat[np.random.choice(range(mutmat.shape[0]), int(rs*mutnum.shape[0]), replace=False)]
    return get_branchlen(mutmat)

def get_data_from_tree(file):
    tree = Phylo.read(file, format='newick')