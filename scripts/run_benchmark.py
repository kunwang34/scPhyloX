import os


sl_script = [
'#!/bin/bash',
'#SBATCH -J mutrate_est',
'#SBATCH -p all',
'#SBATCH -N 1',
'#SBATCH -n 4',
'#SBATCH --mem=1G',
'#SBATCH -t 0',
'#SBATCH -o oe/%x-%j.log ',
'#SBATCH -e oe/%x-%j.err' ]

for i in os.listdir('/home/wangkun/scPhyloX/datasets/cellnumber_sample1'):
    if not i[-2:] == 'gz':
        continue
    with open('./sscript', 'w') as f:
        f.write('\n'.join(sl_script))
        f.write(f'\npython cell_number_and_sample_tumor.py -f {i}')
    os.system('sbatch sscript')
