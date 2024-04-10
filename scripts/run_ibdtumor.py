import os
path = '../datasets/1.IBD_T_Monoclonal'
files = os.listdir(path)
files = [i for i in files if i[-1]=='y']

sl_script = [
'#!/bin/bash',
'#SBATCH -J tumor_est',
'#SBATCH -p all',
'#SBATCH -N 1',
'#SBATCH -n 5',
'#SBATCH --mem=1G',
'#SBATCH -t 0',
'#SBATCH -o oe/%x-%j.log ',
'#SBATCH -e oe/%x-%j.err' ]


for i in files:

    with open('./sscript', 'w') as f:
        f.write('\n'.join(sl_script))
        f.write(f'\npython IBD_tumor_est.py -p {path} -f {i}')

    os.system('sbatch sscript')