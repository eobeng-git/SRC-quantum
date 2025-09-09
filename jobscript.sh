#!/usr/bin/env bash

#SBATCH --nodes 1

cd /NFSHOME/eobeng/src.disim/qml_multispectral
source /NFSHOME/eobeng/miniconda3/etc/profile.d/conda.sh   # <-- Add this line
conda activate /NFSHOME/eobeng/miniconda3/envs/qml_for_multispectral_eo_data

python3 experiment_pqk.py && python3 experiment_classification.py

