
import os
path = '../datasets/HSC_lifespan/'
files = os.listdir(path)
files = [i for i in files if i[0]=='f']


ages = {'AX001':63, 'KX001':29, 'KX002':38, 'KX003':81, 'KX004':77, 'KX007':75, 'KX008':76, 'SX001':48}

for file in files:
    ff = path+file+'/'+os.listdir(path+file)[0]
    sl_script = [
    '#!/bin/bash',
    '#SBATCH -J HSC_est',
    '#SBATCH -p all',
    '#SBATCH -N 1',
    '#SBATCH -n 5',
    '#SBATCH --mem=1G',
    '#SBATCH -t 0',
    f'#SBATCH -o oe/%x-%j{file}.log ',
    f'#SBATCH -e oe/%x-%j{file}.err' ]
    
    with open('./sscript', 'w') as f:
        f.write('\n'.join(sl_script))
        f.write(f"\npython hsc_est.py -f {ff} -t {ages[file.split('_')[-1]]}")

    os.system('sbatch sscript')
    
