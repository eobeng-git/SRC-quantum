"""
Code for experiments from the paper:
`Potential of Quantum Machine Learning for Processing Multispectral Earth Observation Data'
This version has been anonymized for the purpose of peer review and may not be used for any other purpose. 
The code for the experiments will be publicly available under an open license - links to the repository 
will be found in the paper.
-------------------------------
print results in latex table format
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations

class_pairs = [(62, 73), (62, 82), (62, 102), (73, 82), (73, 102), (82, 102)]
classifiers = ["dtree", "knn", "nb", "svc"]


def load_aggregated_scores(
    class_pair,
    scenario_name,
    n_instances=10,
    path="results/",
    metric="accuracy",
):
    """
    Loads, concatenates, and aggregates train/test accuracy for a specific scenario.
    e.g. scenario_name could be 'no_pqk_relabel' or 'pqk_relabel'.

    Returns a DataFrame with columns:
        ['train_score_mean','test_score_mean','train_score_std','test_score_std']
    indexed by classifier.
    """
    pair_str = f"{class_pair[0]}_{class_pair[1]}_{scenario_name}"
    all_df = []
    for i in range(n_instances):
        fname = Path(path) / f"{pair_str}_{metric}_{i}.pkl"
        if fname.exists():
            df = pd.read_pickle(fname)
            all_df.append(df)
        else:
            print(f"fname {fname} does not exist!")
    if not all_df:
        return None

    combined = pd.concat(all_df, ignore_index=True)
    means = combined.groupby("cls_name")[["train_score", "test_score"]].mean()
    stds = combined.groupby("cls_name")[["train_score", "test_score"]].std()

    out = pd.concat([means, stds], axis=1)
    out.columns = [
        "train_score_mean",
        "test_score_mean",
        "train_score_std",
        "test_score_std",
    ]
    return out


def get_results(
    n_instances=10,
    path="results/",
    file_original="no_pqk_original_label",
    file_pqk="pqk_original_label",
    metric="accuracy",
):
    """
    Reads all class-pair experiments for relabeled data (no_pqk_relabel vs pqk_relabel)
    and returns a DataFrame with 10 columns:
        class_pair
        Classifier
        train_original_mean
        train_original_std
        train_pqk_mean
        train_pqk_std
        test_original_mean
        test_original_std
        test_pqk_mean
        test_pqk_std

    All these columns remain numeric (where applicable) so you can do arithmetic.
    """
    rows = []

    for cp in class_pairs:
        # Load aggregated results for Original vs PQK.
        original_scores = load_aggregated_scores(
            cp, file_original, n_instances, path, metric=metric
        )
        pqk_scores = load_aggregated_scores(
            cp, file_pqk, n_instances, path, metric=metric
        )
        if (original_scores is None) or (pqk_scores is None):
            continue

        for clf in classifiers:
            if clf not in original_scores.index or clf not in pqk_scores.index:
                continue

            row = {
                "class_pair": f"{cp[0]}_{cp[1]}",
                "Classifier": clf,
                # Original scenario
                "train_original_mean": original_scores.loc[clf, "train_score_mean"],
                "train_original_std": original_scores.loc[clf, "train_score_std"],
                "test_original_mean": original_scores.loc[clf, "test_score_mean"],
                "test_original_std": original_scores.loc[clf, "test_score_std"],
                # PQK scenario
                "train_pqk_mean": pqk_scores.loc[clf, "train_score_mean"],
                "train_pqk_std": pqk_scores.loc[clf, "train_score_std"],
                "test_pqk_mean": pqk_scores.loc[clf, "test_score_mean"],
                "test_pqk_std": pqk_scores.loc[clf, "test_score_std"],
            }
            rows.append(row)

    df_new = pd.DataFrame(
        rows,
        columns=[
            "class_pair",
            "Classifier",
            "train_original_mean",
            "train_original_std",
            "train_pqk_mean",
            "train_pqk_std",
            "test_original_mean",
            "test_original_std",
            "test_pqk_mean",
            "test_pqk_std",
        ],
    )

    return df_new


def format_results(df_new: pd.DataFrame) -> pd.DataFrame:
    """
    Converts a DataFrame from present_all_relabel_results_new_format (with numeric columns)
    to the "old" format DataFrame. If 'class_pair' is missing, the output DataFrame omits it.

    Columns included:
        - Always: Classifier, Train_Original, Train_PQK, Test_Original, Test_PQK
        - Conditionally: class_pair (if present in df_new)

    The last four columns are formatted as "mean ± std".
    """
    # Required numeric columns
    required_cols = {
        "Classifier",
        "train_original_mean",
        "train_original_std",
        "train_pqk_mean",
        "train_pqk_std",
        "test_original_mean",
        "test_original_std",
        "test_pqk_mean",
        "test_pqk_std",
    }

    # Ensure required columns exist
    missing_cols = required_cols - set(df_new.columns)

    if missing_cols:
        raise ValueError(f"DataFrame missing required columns: {missing_cols}")

    include_class_pair = "class_pair" in df_new.columns

    # Construct rows
    old_format_rows = []
    for _, row in df_new.iterrows():
        new_row = {
            "Classifier": row["Classifier"],
            "Train_Original": f"${row['train_original_mean']:.2f} \pm {row['train_original_std']:.2f}$",
            "Train_PQK": f"${row['train_pqk_mean']:.2f} \pm {row['train_pqk_std']:.2f}$",
            "Test_Original": f"${row['test_original_mean']:.2f} \pm {row['test_original_std']:.2f}$",
            "Test_PQK": f"${row['test_pqk_mean']:.2f} \pm {row['test_pqk_std']:.2f}$",
        }
        if include_class_pair:
            new_row["class_pair"] = row["class_pair"]

        old_format_rows.append(new_row)

    # Define output columns
    columns = [
        "class_pair",
        "Classifier",
        "Train_Original",
        "Train_PQK",
        "Test_Original",
        "Test_PQK",
    ]
    if not include_class_pair:
        columns.remove("class_pair")

    return pd.DataFrame(old_format_rows, columns=columns)


def get_latex_table(
    path="results/",
    file_original="no_pqk_relabel",
    file_pqk="pqk_relabel",
    metric="accuracy",
):
    """Get latex table for a given scenario"""
    df_intermediate = get_results(
        n_instances=10,
        path=path,
        file_original=file_original,
        file_pqk=file_pqk,
        metric=metric,
    )

    df_intermediate = (
        df_intermediate.iloc[:, 1:].groupby("Classifier").mean().reset_index()
    )
    df_formatted = format_results(df_intermediate)

    relabel_res = df_formatted
    print(df_formatted.to_latex(float_format="%.2f"))


if __name__ == "__main__":
    get_latex_table(
        path="results/",
        file_original="no_pqk_relabel",
        file_pqk="pqk_relabel",
        metric="accuracy",
    )
    get_latex_table(
        path="results/",
        file_original="no_pqk_original_label",
        file_pqk="pqk_original_label",
        metric="accuracy",
    )
