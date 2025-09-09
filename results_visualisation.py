"""
Code for experiments from the paper:
`Potential of Quantum Machine Learning for Processing Multispectral Earth Observation Data'
This version has been anonymized for the purpose of peer review and may not be used for any other purpose. 
The code for the experiments will be publicly available under an open license - links to the repository 
will be found in the paper.
-------------------------------
RGB data visualisation
Warning: seaborn package required. 
"""

import pandas as pd
import numpy as np
from sklearn import datasets
from classification_experiment import perform_experiment
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def replace_cls_names(df):
    """Replace values in 'cls_name' according to mapping if column exists."""
    mapping = {"svc": "SVC", "knn": "kNN", "dtree": "DT", "nb": "NB"}
    if "cls_name" in df.columns:
        df["cls_name"] = df["cls_name"].replace(mapping)
    return df


def present_results(name, n_instances=10, path="results/", scoring="accuracy"):
    """
    presents experiment visualisation for single named experiment
    """
    found = []
    results = []

    for i_instance in range(n_instances + 1):
        fname = Path(f"{path}{name}_{scoring}_{i_instance}.pkl")
        try:
            df_result = pd.read_pickle(fname)
            found.append(i_instance)
            results.append(df_result)
        except:
            print(f"fname {fname} does not exist!")
            pass

    if len(results) > 0:
        final_results = pd.concat(results, axis=0)
        final_results = replace_cls_names(final_results)
        print(final_results)
        means = final_results.groupby("cls_name").mean()
        stds = final_results.groupby("cls_name").std()
        scores = pd.concat([means, stds], axis=1)
        scores.columns = [
            "train_score_mean",
            "test_score_mean",
            "train_score_std",
            "test_score_std",
        ]
        print(f"Name: {name}, instances: {len(found)}")
        print(f"found iterations: {found}")
        print(scores)
        plt.rcParams.update({"font.size": 12})
        plot = sns.boxplot(
            data=final_results, y="test_score", x="cls_name", palette="Set1"
        )
        plot.set_ylim(0.35, 0.95)
        plt.xlabel("Classifier")
        plt.ylabel("Accuracy")
        plt.tight_layout()
        plt.savefig(name + ".eps")
    else:
        print("No results = no visualisation :(")


if __name__ == "__main__":
    name = "82_102_no_pqk_relabel"
    n_instances = 10
    path = "results/"  # your path, if it differs fro this
    present_results(name, n_instances=n_instances, path=path)
