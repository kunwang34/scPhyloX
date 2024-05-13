from scipy.stats import pearsonr
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import numpy as np
import warnings

def corr_plot(x, y, ax, stats='pearson', r0_x=None, r0_y=None, r1_x=None, r1_y=None, line='fit', alternative='two-sided', fontsize=10):
    '''
    Draw a scatter plot of the two sets of data and show their correlation coefficients
    
    Args:
        x: 
            data1
        y: 
            data2
        ax: 
            axes to draw scatter on
        stats: 
            pearson or spearman
        r0_x, r0_y, r1_x, r1_y: 
            locations to label the correlation coefficient and the p-value
        fontsize:
            fontsize
    Return:
        matplotlib.axes
    '''
    stats = stats.lower()

    ax.scatter(x, y, alpha=0.6, s=70)
    if line == 'fit':
        a, b = np.polyfit(x, y, deg=1)
        y_est = a * np.linspace(min(x),max(x),60) + b
        ax.plot(np.linspace(min(x),max(x),60), y_est, '-', c='k')
    elif line == 'diag':
        mi, ma = min(min(x),min(y)), max(max(x), max(y))
        ax.plot((mi, ma), (mi, ma), '-', c='k')
    
    dx = ax.get_xlim()[1]-ax.get_xlim()[0]
    dy = ax.get_ylim()[1]-ax.get_ylim()[0]
    
    if r0_x is None:
        r0_x = ax.get_xlim()[0]+dx*0.05
    if r0_y is None:
        r0_y = ax.get_ylim()[1]-dy*0.05
    if r1_x is None:
        r1_x = ax.get_xlim()[0]+dx*0.05
    if r1_y is None:
        r1_y = ax.get_ylim()[1]-dy*0.15 
        
    if stats == 'pearson':
        r, pval = pearsonr(x, y, alternative=alternative)
        ax.text(r0_x, r0_y, r"Pearson's $r={:.2g}$".format(r), fontsize=fontsize)
    else:
        r, pval = spearmanr(x, y, alternative=alternative)
        ax.text(r0_x, r0_y, r"Spearman's $\rho={:.2g}$".format(r), fontsize=fontsize)
    if pval == 0:
        ax.text(r1_x, r1_y, r'$P<10^{-100}$', fontsize=fontsize)
    elif pval >= 0.01:
        ax.text(r1_x, r1_y, r'$P={:.2g}$'.format(pval), fontsize=fontsize)
    else:
        try:
            ax.text(r1_x, r1_y, r'$P={}\times 10^{}$'.format(*r'{:.2e}'.format(pval).split('e')).replace('^', '^{').replace('$', '}$')[1:], fontsize=fontsize)
        except:
            ax.text(r1_x, r1_y, r'$P={:.4g}$'.format(pval), fontsize=fontsize)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return ax


def colless_index(tree):
    nodes = tree.get_nonterminals()
    ci = []
    binary = 1
    for i in nodes:
        if len(i.clades) > 2:
            binary = 0
        ci.append(np.abs(len(i.clades[0].get_terminals())-len(i.clades[1].get_terminals())))
    # if not binary:
    #     warnings.warn('Not binary tree')
    return ci

def colless_index_corrected(tree):
    ci = np.sum(colless_index(tree))
    n = len(tree.get_terminals())
    return 2*ci/(n-1)/(n-2)
    
    
def ext_gen(cell):
    if cell == '0':
        return -1
    return int(cell.split('_')[0][1:])

def ext_cid(cell):
    if cell == '0':
        return 0
    return int(cell.split('_')[1][:-1])

def reconstruct(lineage_info, sel_cells, file_name):
    new_keep = deepcopy(sel_cells)
    sample_index = deepcopy(sel_cells)
    while new_keep:
        # print(new_keep)
        new_parents = []
        for i in new_keep:
            curr_gen = int(ext_gen(i))
            if curr_gen == 0:
                continue
            new_parents.append(f'<{curr_gen-1}_{lineage_info[i]}>')
        while '0' in new_parents:
            new_parents.remove('0')
            
        new_keep = deepcopy(new_parents)
        sample_index.extend(new_parents)
    sample_index = np.unique(sample_index)
    # sample_index = np.insert(sample_index, 0, 0)
    data = pd.DataFrame(index=sample_index, columns=['info', 'generation','cell_id', 'parent_id'])
    # return data
    data["info"] = [
        "<{:d}_{:d}>:{}".format(
            int(ext_gen(i)),
            int(ext_cid(i)),
            1,
        )
        for i in data.index
    ]
    
    data['generation'] = [ext_gen(i) for i in data.index]
    data['parent_id'] = [lineage_info[i] for i in data.index]
    data['cell_id'] = [ext_cid(i) for i in data.index]
    # states = [
    #     "<{:d}_{:d}>:{:d}".format(
    #         int(data.loc[i]["generation"]),
    #         int(data.loc[i]["cell_id"]),
    #         int(data.loc[i]["state"]),
    #     )
    #     for i in data.index
    # ]
    gen = data.generation.max()
    tree = []
    # data.loc[0] = ['<-1_0>:1', '-1', '0', '0']
    while gen:        
        for pid in set(data[data.generation == gen].parent_id.to_numpy()):
            pair = data[
                np.all(list(zip(data.generation == gen, data.parent_id == pid)), axis=1)
            ]
            parent_index = data[
                np.all(
                    list(zip(data.generation == gen - 1, data.cell_id == pid)), axis=1
                )
            ].index[0]
            oi = data.loc[parent_index, "info"]
            if pair.shape[0] == 2:
                ni = "({}, {}){}".format(pair.iloc[0]["info"], pair.iloc[1]["info"], oi)
            else:
                ni = "({}){}".format(pair.iloc[0]["info"], oi)
            data.loc[parent_index, "info"] = ni
            data = data.drop(index=pair.index)
        gen -= 1
    with open(file_name, 'w') as f:
        f.write(f"({','.join(tr['info'])})")
    return data