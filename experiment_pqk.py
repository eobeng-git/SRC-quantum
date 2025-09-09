"""
Code for experiments from the paper:
`Potential of Quantum Machine Learning for Processing Multispectral Earth Observation Data'
This version has been anonymized for the purpose of peer review and may not be used for any other purpose. 
The code for the experiments will be publicly available under an open license - links to the repository 
will be found in the paper.
-------------------------------
PQK experiment
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn.model_selection as sklms
import sklearn.preprocessing as sklprep
import skopt
import pickle
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.metrics.pairwise import euclidean_distances
from datetime import datetime
from itertools import combinations

from pqk import PqkTransform
from classification_experiment import perform_experiment

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def filter_2cl(x, y, class1, class2):
    keep = (y == class1) | (y == class2)
    x, y = x[keep], y[keep]
    y = y == class1
    class1_sample_size = len(y[y == 1])
    class2_sample_size = len(y[y == 0])
    data_set_size = (
        class1_sample_size
        if class1_sample_size < class2_sample_size
        else class2_sample_size
    )
    print(data_set_size)
    print("class1 sample size: ", class1_sample_size)
    print("class2 sample size: ", class2_sample_size)
    return x, y


def balance_2cl(X, Y, dataset_size, random_state):
    rng = np.random.default_rng(random_state)
    index = np.arange(len(Y))
    indices = np.unique(
        np.concatenate(
            [rng.permutation(index[Y == c])[:dataset_size] for c in np.unique(Y)]
        )
    )
    Y = Y[indices]
    X = X[indices]

    return X, Y


def plot_fig(score_matrix, pqk_feature, relable):

    path = "../results/images/"
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

    fig, ax = plt.subplots()
    if pqk_feature == True and relable == True:
        title = "Binary classification on relabeled data with PQK feature"
        file_name = "boxplot_pqk_on_relabled_data.eps"
    elif pqk_feature == False and relable == True:
        title = "Binary classification on relabeled data without PQK feature"
        file_name = "boxplot_without_pqk_on_relabled_data.eps"
    elif pqk_feature == True and relable == False:
        title = "Binary classification on original data with PQK feature"
        file_name = "boxplot_pqk_on_original_data.eps"
    elif pqk_feature == False and relable == False:
        title = "Binary classification on original data without PQK feature"
        file_name = "boxplot_without_pqk_on_original_data.eps"

    # basic plot
    bplot = ax.boxplot(
        (score_matrix[0], score_matrix[1], score_matrix[2], score_matrix[3]),
        patch_artist=True,
        labels=["SVC", "KNN", "Naive Bayes", "Decision Tree"],
    )
    # ax.boxplot(score_matrix, positions=['SVC', 'KNN', 'Naïve Bayes', 'Decision Tree'])
    ax.set_title(title)
    ax.set_ylabel("CV Accuracy (5- fold)")
    ax.set_ylim(0.55, 0.93)
    colors = ["blue", "red", "lime", "magenta"]

    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)
    # ax.set_facecolor(colors = ['blue', 'red', 'green','pink'])
    plt.savefig(path + file_name)
    plt.show()


def get_stilted_dataset(S, V, S_2, V_2, lambdav=1.1):
    """Prepare new labels that maximize geometric distance between kernels."""
    S_diag = tf.linalg.diag(S**0.5)
    S_2_diag = tf.linalg.diag(S_2 / (S_2 + lambdav) ** 2)
    scaling = S_diag @ tf.transpose(V) @ V_2 @ S_2_diag @ tf.transpose(V_2) @ V @ S_diag

    # Generate new lables using the largest eigenvector.
    _, vecs = tf.linalg.eig(scaling)
    new_labels = tf.math.real(
        tf.einsum("ij,j->i", tf.cast(V @ S_diag, tf.complex64), vecs[-1])
    ).numpy()
    # Create new labels and add some small amount of noise.
    final_y = new_labels > np.median(new_labels)
    noisy_y = final_y ^ (np.random.uniform(size=final_y.shape) > 0.95)
    return noisy_y


def compute_kernel_matrix(vecs, gamma):
    """Computes d[i][j] = e^ -gamma * (vecs[i] - vecs[j]) ** 2"""
    scaled_gamma = gamma / (
        tf.cast(tf.gather(tf.shape(vecs), 1), tf.float32) * tf.math.reduce_std(vecs)
    )
    return scaled_gamma * tf.einsum("ijk->ij", (vecs[:, None, :] - vecs) ** 2)


def get_spectrum(datapoints, gamma=1.0):
    """Compute the eigenvalues and eigenvectors of the kernel of datapoints."""
    KC_qs = compute_kernel_matrix(datapoints, gamma)
    S, V = tf.linalg.eigh(KC_qs)
    S = tf.math.abs(S)
    return S, V


def compute_svm_gamma(x_train, y_train):
    # Hyperparameter optimization of SVM to find C and gamma
    clf = svm.SVC()
    cv = sklms.KFold()
    params = {"C": (1e-6, 100.0, "log-uniform"), "gamma": (1e-6, 100.0, "log-uniform")}
    search = skopt.BayesSearchCV(estimator=clf, search_spaces=params, n_jobs=-1, cv=cv)
    search.fit(x_train, y_train)
    print(f"best score: {search.best_score_}")
    print(f"best params: {search.best_params_}")
    gamma_best = search.best_params_["gamma"]
    return gamma_best


def test_gamma(X):
    """
    Computes the Smola heuristics for gamma
    see https://arxiv.org/abs/2111.02164
    Gives you three values of gamma  - one of them should be ,,right'' for your data
    """
    for p in [0.1, 0.5, 0.9]:
        distances = euclidean_distances(X, squared=False)
        distances = distances[np.triu_indices(len(X), k=1)]
        s_lambda = 1.0 / (np.quantile(distances, p))
        gamma = s_lambda**2
        print(f"p value: {p},", f"gamma: {gamma}")


def gen_pqk_relabeled_data(X, y, run):

    # Due to memory issue with GPU wit more data points
    max_n = 1000

    # Size of training set
    train_size = 0.8

    # Size of test set
    test_size = 0.2

    train_dataset_size = int(max_n * train_size)
    test_dataset_size = int(max_n * test_size)

    n_trotter = 10
    dataset_dim = 4

    rng = np.random.default_rng(seed=12345)
    random_rots = rng.uniform(-2, 2, size=(dataset_dim + 2, 3))

    # Split it into 'training' and 'test' parts
    x_train, x_test, y_train, y_test = sklms.train_test_split(
        X, y, train_size=train_size, test_size=test_size, random_state=run, shuffle=True
    )

    # Standardize features by removing the mean and scaling to unit variance.
    scaler = sklprep.StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # x_train, x_test, y_train, y_test = x_train[:800], x_test[:200], y_train[:800], y_test[:200]
    x_train, y_train = balance_2cl(
        x_train, y_train, dataset_size=train_dataset_size, random_state=run
    )
    x_test, y_test = balance_2cl(
        x_test, y_test, dataset_size=test_dataset_size, random_state=run
    )

    print("y_train class1 sample size: ", len(y_train[y_train == 1]))
    print("y_train class2 sample size: ", len(y_train[y_train == 0]))
    print("y_test class1 sample size: ", len(y_test[y_test == 1]))
    print("y_test class2 sample size: ", len(y_test[y_test == 0]))

    # Downscale images using PCA
    pca = PCA(n_components=dataset_dim)
    x_train_reduced = pca.fit_transform(x_train)
    x_test_reduced = pca.transform(x_test)

    print("gamma with reduced features")
    gamma_reduced_feature = compute_svm_gamma(x_train_reduced, y_train)
    test_gamma(x_train_reduced)

    # For the time being I just sellect gamma from reduced features. TODO: how to sellect best gamme?
    gamma_best = gamma_reduced_feature

    # Obtain PQK features
    pqk = PqkTransform(random_rots, n_trotter)
    x_train_pqk = pqk.fit_transform(x_train_reduced)
    x_test_pqk = pqk.transform(x_test_reduced)

    # print('shape:', np.shape(x_train_pqk))
    # print('new shape',tf.shape(tf.concat([x_train_pqk, x_test_pqk], 0)))

    S_pqk, V_pqk = get_spectrum(
        tf.cast(tf.concat([x_train_pqk, x_test_pqk], 0), tf.float32)
    )

    S_original, V_original = get_spectrum(
        tf.cast(tf.concat([x_train_reduced, x_test_reduced], 0), tf.float32),
        gamma=gamma_best,
    )

    y_relabel = get_stilted_dataset(S_pqk, V_pqk, S_original, V_original)
    y_original = np.concatenate((y_train, y_test), axis=0)

    x_pqk = np.concatenate((x_train_pqk, x_test_pqk), axis=0)
    x_reduced = np.concatenate((x_train_reduced, x_test_reduced), axis=0)

    return x_reduced, x_pqk, y_original, y_relabel


def main():

    PATH = "data/"
    result_path = "results/"

    # Check if result directory exists
    isExist = os.path.exists(result_path)
    if not isExist:
        os.makedirs(result_path)

    # Load SpacePenny dataset
    data = np.load(PATH + "sentinel.npz")
    raw_data = data["bands"]
    raw_gt = data["classes"]

    # data class labels
    classes_names = {
        62: "Artificial surfaces and constructions",
        73: "Cultivated areas",
        82: "Broadleaf tree cover",
        102: "Herbaceous vegetation",
    }

    start_time = datetime.now()

    # generate class pairs
    class_pair_list = combinations(classes_names, 2)

    for class_pair in class_pair_list:

        print(f"Running experiment for class pair ({class_pair[0]}, {class_pair[1]})")
        # Filter data of given classes
        X, y = filter_2cl(raw_data, raw_gt, class_pair[0], class_pair[1])

        for run in range(10):

            # Generates a new relabeled dataset after each call
            x_reduced, x_pqk, y_original, y_relabel = gen_pqk_relabeled_data(X, y, run)

            data = {
                "ori_features": x_reduced,
                "pqk_features": x_pqk,
                "original_label": y_original,
                "relabel": y_relabel,
            }

            name = f"{result_path}{str(class_pair[0])}_{str(class_pair[1])}_{run}_cache.pkl"
            with open(name, "wb") as file:
                pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)

    time_elapsed = datetime.now() - start_time

    print("Total execution time (hh:mm:ss.ms) {}".format(time_elapsed))


if __name__ == "__main__":
    main()
