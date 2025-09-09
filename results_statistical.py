"""
Code for experiments from the paper:
`Potential of Quantum Machine Learning for Processing Multispectral Earth Observation Data'
This version has been anonymized for the purpose of peer review and may not be used for any other purpose. 
The code for the experiments will be publicly available under an open license - links to the repository 
will be found in the paper.
-------------------------------
Statistical analysis using baycomp
Warning: note that pystan==3.4.0 works, and for this python 3.8 is recommended. Therefore, it is recommended to use a separate conda environment.
"""

import pandas as pd
from pathlib import Path
from baycomp import two_on_multiple, two_on_single
import matplotlib.pyplot as plt
import numpy as np
import sys


def load_all_scores(
    class_pairs, result_string, path="temp", metric="accuracy", n_instances=10
):
    """Load and concatenate all runs for each class pair into a single DataFrame."""
    dfs = []
    for cp in class_pairs:
        pair_str = f"{cp[0]}_{cp[1]}"
        for run in range(n_instances):
            fname = Path(path) / f"{pair_str}_{result_string}_{metric}_{run}.pkl"
            if fname.exists():
                df = pd.read_pickle(fname)
                df["run"] = run
                df["class_pair"] = pair_str
                dfs.append(df)
            else:
                print("no", fname)

    if not dfs:
        print("No files found.")
        return None

    full_df = pd.concat(dfs, ignore_index=True)
    return full_df


def query(
    result_string="pqk_original_label",
    cls1="knn",
    cls2="svc",
    path="results",
    metric="accuracy",
    rope=0.02,
    nsamples=50000,
):
    class_pairs = [(62, 73), (62, 82), (62, 102), (73, 82), (73, 102), (82, 102)]
    full_df = load_all_scores(
        class_pairs,
        path=path,
        metric=metric,
        n_instances=10,
        result_string=result_string,
    )

    full_df = full_df[["run", "class_pair", "cls_name", "train_score", "test_score"]]

    cls1_df = full_df[full_df["cls_name"] == cls1]
    cls2_df = full_df[full_df["cls_name"] == cls2]

    cls1_df = cls1_df.groupby("class_pair")["test_score"].apply(list)
    cls2_df = cls2_df.groupby("class_pair")["test_score"].apply(list)

    assert (cls1_df.index == cls2_df.index).all()

    cls1_df = np.array(cls1_df.tolist())
    cls2_df = np.array(cls2_df.tolist())

    (p_left, p_rope, p_right), fig = two_on_multiple(
        cls1_df,
        cls2_df,
        rope=rope,
        plot=True,
        nsamples=nsamples,
        runs=5,
        names=[cls1, cls2],
    )
    print(p_left, p_rope, p_right)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    query(
        result_string="pqk_original_label",
        cls1="knn",
        cls2="svc",
        path="results",
        metric="accuracy",
    )
