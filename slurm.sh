#!/bin/bash

#SBATCH -n 8
#SBATCH --mem-per-cpu=4000
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err
#SBATCH --time=4:00:00
#SBATCH --gpus=1

module load gcc/8.2.0 python_gpu/3.8.5 eth_proxy
export PYTHONPATH=/cluster/home/zicfan/.local/lib/python3.8/site-packages:$PYTHONPATH
export HMR_API_KEY=""
export HMR_WORKSPACE=""
source $HOME/.local/bin/virtualenvwrapper.sh
workon mp_env
python scripts/train.py --trainsplit fulltrain --valsplit none
