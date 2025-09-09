"""
Code for experiments from the paper:
`Potential of Quantum Machine Learning for Processing Multispectral Earth Observation Data'
This version has been anonymized for the purpose of peer review and may not be used for any other purpose. 
The code for the experiments will be publicly available under an open license - links to the repository 
will be found in the paper.
-------------------------------
Classification experiment
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import combinations
from classification_experiment import perform_experiment

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


classes_names = {
    62: "Artificial surfaces and constructions",
    73: "Cultivated areas",
    82: "Broadleaf tree cover",
    102: "Herbaceous vegetation",
}


def main(scoring="accuracy", n_jobs=20, result_path="results/"):
    """
    Perform classification experiments from cached datafor all class pairs
    """
    os.makedirs(result_path, exist_ok=True)

    class_pair_list = list(combinations(classes_names, 2))

    start_time = datetime.now()

    for class_pair in class_pair_list:

        for run in range(10):
            cache_path = f"results/{class_pair[0]}_{class_pair[1]}_{run}_cache.pkl"
            if not os.path.exists(cache_path):
                print(f"Cache not found: {cache_path}")
                continue

            cached_data = pd.read_pickle(cache_path)
            x_reduced = cached_data["ori_features"]
            x_pqk = cached_data["pqk_features"]
            y_original = cached_data["original_label"]
            y_relabel = cached_data["relabel"]

            print("run", run)
            perform_experiment(
                X=x_reduced,
                y=y_relabel,
                instance_index=run,
                name=f"{class_pair[0]}_{class_pair[1]}_no_pqk_relabel",
                scoring=scoring,
                path=result_path,
                n_jobs=n_jobs,
            )

            perform_experiment(
                X=x_pqk,
                y=y_relabel,
                instance_index=run,
                name=f"{class_pair[0]}_{class_pair[1]}_pqk_relabel",
                scoring=scoring,
                path=result_path,
                n_jobs=n_jobs,
            )

            perform_experiment(
                X=x_reduced,
                y=y_original,
                instance_index=run,
                name=f"{class_pair[0]}_{class_pair[1]}_no_pqk_original_label",
                scoring=scoring,
                path=result_path,
                n_jobs=n_jobs,
            )

            perform_experiment(
                X=x_pqk,
                y=y_original,
                instance_index=run,
                name=f"{class_pair[0]}_{class_pair[1]}_pqk_original_label",
                scoring=scoring,
                path=result_path,
                n_jobs=n_jobs,
            )

    time_elapsed = datetime.now() - start_time
    print(f"Total execution time (hh:mm:ss.ms) {time_elapsed}")


if __name__ == "__main__":
    scores = ["accuracy", "matthews_corrcoef", "recall", "f1", "balanced_accuracy"]
    main(scoring=scores[0])
