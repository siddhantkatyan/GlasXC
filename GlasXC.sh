#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -n 10
#SBATCH -p long
#SBATCH --mem-per-cpu=3000
#SBATCH --time=04-00:00:00
#SBATCH --mail-type=END

clear

#conda activate DL

source ~/anaconda3/etc/profile.d/conda.sh

conda activate DL

#conda list

./train_GlasXC_with_args.sh

#python hello.py

#conda deactivate DL

#conda deactivate DL