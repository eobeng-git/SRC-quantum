# Potential of Quantum Machine Learning for Processing Multispectral Earth Observation Data

This repository allows to replicate experiments from the paper:
`Potential of Quantum Machine Learning for Processing Multispectral Earth Observation Data'
This version has been anonymized for the purpose of peer review and may not be used for any other purpose. The code for the experiments will be publicly available under an open license - links to the repository will be found in the paper.

# Abstract
Quantum computers with hundreds of noisy qubits are already available for the research community. 
They have the potential to run complex quantum computations well beyond the computational 
capacity of any classical device. It is natural to ask the question, 
what application these devices could be useful for? Land Use and Land Cover 
classification of multispectral Earth observation data collected from the earth 
observation satellite mission is one such problem that is hard for classical 
methods due to its unique characteristics. In this work, we compare the 
performance of several classical machine learning algorithms on the stilted re-labeled 
dataset of the Copernicus Sentinel-2 mission, when the algorithm has access to 
Projected Quantum Kernel (PQK) features. We show that the classification accuracy 
increases drastically when the model has access to PQK features. We then naively 
study the performance of these algorithms with and without access to PQK features 
on the original Copernicus Sentinel-2 mission data set. This study provides key 
evidence that shows the potential of quantum machine learning methods for Earth 
Observation data.

# Licence
This version has been anonymized for the purpose of peer review and may not be used for any other purpose. The code for the experiments will be publicly available under an open license - links to the repository will be found in the paper.


# Installation
1. use environment.yml to create python env
    - note that we use baycomp (https://github.com/janezd/baycomp), which requires pystan==3.4.0  and for this python 3.8 is recommended. Consider a separate environment.
2. run experiment_pqk.py and experiment_classification.py
3. results_* files demonstrate data analysis


# Files
    - classification_experiment/alg_experiment.py: classification experiment with 2-stage stratified CV
    - experiment_pqk.pu - PQK experiment
    - epxeriment_classification.py - Classification experiment
    - results_data_visalisation.py - Dataset RGB visualisation
    - results_table.py - table generation
    - results_statistical.py - statistical analysis
    - results_visualisation.py - plot generation

