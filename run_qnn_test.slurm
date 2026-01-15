#!/bin/bash
#SBATCH --job-name=qnn-test
#SBATCH --nodes=1
#SBATCH --time=240:00:00
#SBATCH --output=qnn_test_%j.out
#SBATCH --error=qnn_test_%j.err
#SBATCH --mail-user=emmanuel.obeng@student.univaq.it
#SBATCH --mail-type=ALL


source ~/.bashrc
conda activate qnn-310
cd /NFSHOME/eobeng/src.disim

python alg_experiment_qnn_fixed.py
