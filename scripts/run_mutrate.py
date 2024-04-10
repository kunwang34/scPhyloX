import os

flys = ['L5', 'L6']
organs = 'Br,Ey,Fb,L1,L2,L3,Mp,Sg,Wg'.split(',')

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


for i in flys:
    for j in organs:
        with open('./sscript', 'w') as f:
            f.write('\n'.join(sl_script))
            f.write(f'\npython mutation_rate_estimate.py -f {i} -o {j}')

        os.system('sbatch sscript')