"""
Classification experiment with classical models + safe PennyLane QNN
Flattens 2D image data to 1D tabular for standard CV
Uses pure default.qubit backend (no JAX/Lightning/AVX2 required)
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
import pennylane as qml
from pennylane import numpy as np_pennylane

warnings.simplefilter(action="ignore", category=FutureWarning)

# =============================================================================
# CLASSICAL MODELS
# =============================================================================

MODELS = {
    "svc": {
        "estimator": SVC(),
        "param_grid": {"C": [0.01, 0.1, 1, 10, 100], "gamma": [0.01, 0.1, 1, 10, 100]},
    },
    "knn": {
        "estimator": KNeighborsClassifier(),
        "param_grid": {
            "n_neighbors": [1, 3, 5, 7, 11, 15],
            "weights": ["uniform", "distance"],
        },
    },
    "nb": {"estimator": GaussianNB(), "param_grid": {}},
    "dtree": {
        "estimator": DecisionTreeClassifier(),
        "param_grid": {
            "max_depth": [None, 2, 3, 5, 10],
            "min_samples_split": [2, 3, 4],
        },
    },
}

# =============================================================================
# PENNYLANE QNN (pure default.qubit - CPU-safe)
# =============================================================================

n_qubits = 4  # small for speed on your CPU
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="autograd")
def qnn_circuit(inputs, weights):
    qml.AngleEmbedding(inputs[:n_qubits], wires=range(n_qubits))
    for layer in range(2):
        for i in range(n_qubits):
            qml.RX(weights[layer, i, 0], wires=i)
            qml.RY(weights[layer, i, 1], wires=i)
            qml.RZ(weights[layer, i, 2], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
    return qml.expval(qml.PauliZ(0))

class PennyQNN:
    def __init__(self):
        self.weights = np_pennylane.random.random((2, n_qubits, 3))

    def fit(self, X, y):
        pass  # placeholder - training can be added later

    def predict(self, X):
        preds = []
        for x in X:
            val = qnn_circuit(x, self.weights)
            preds.append(1 if val > 0 else 0)  # binary classification example
        return np.array(preds)

    def score(self, X, y):
        """Compatibility method for scikit-learn Pipeline and scoring"""
        preds = self.predict(X)
        return np.mean(preds == y)  # accuracy (fraction correct)

MODELS["penny_qnn"] = {
    "estimator": PennyQNN(),
    "param_grid": {}
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def test_classifier(X_train, y_train, X_test, y_test, k_internal=3, scoring="accuracy", scale_data=True, n_jobs=1, **model):
    estimator = model["estimator"]
    param_grid = model.get("param_grid", {})

    if scale_data:
        pipeline = Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
        param_grid = {f"clf__{k}": v for k, v in param_grid.items()}
    else:
        pipeline = estimator

    if param_grid:
        gscv = GridSearchCV(pipeline, param_grid, cv=k_internal, scoring=scoring, n_jobs=n_jobs, refit=True)
        gscv.fit(X_train, y_train)
        best_model = gscv.best_estimator_
        train_score = gscv.best_score_
    else:
        best_model = pipeline
        best_model.fit(X_train, y_train)
        train_score = best_model.score(X_train, y_train)

    test_score = best_model.score(X_test, y_test)
    return pd.DataFrame({"train_score": [train_score], "test_score": [test_score]})


def perform_experiment(X, y, instance_index=42, name="experiment", k_external=5, k_internal=3, scoring="accuracy", path="results/", scale_data=True, n_jobs=1):
    skf = StratifiedKFold(n_splits=k_external, shuffle=True, random_state=instance_index)
    results = []

    for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y), 1):
        print(f"Fold {fold_idx}/{k_external}")
        fold_results = []

        for model_name, model in MODELS.items():
            print(f"  → {model_name}")
            df = test_classifier(X[train_index], y[train_index], X[test_index], y[test_index],
                                 k_internal=k_internal, scoring=scoring, scale_data=scale_data, n_jobs=n_jobs, **model)
            df["cls_name"] = model_name
            df["fold"] = fold_idx
            fold_results.append(df)

        results.append(pd.concat(fold_results))

    df_results = pd.concat(results, ignore_index=True)

    if path:
        fname = Path(f"{path}{name}_{scoring}_{instance_index}.pkl")
        print(f"Results saved to: {fname}")
        df_results.to_pickle(fname)

    return df_results


# =============================================================================
# MAIN: LOAD DATA & RUN
# =============================================================================

if __name__ == "__main__":
    import numpy as np

    # Load data with correct keys
    data_path = "data/sentinel.npz"
    data = np.load(data_path)

    # Flatten 2D image to 1D tabular format for standard classification
    X = data["bands"].reshape(-1, data["bands"].shape[-1])  # (total_pixels, n_bands)
    y = data["classes"].ravel()  # 1D labels (total_pixels,)

    print(f"Flattened data: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")

    # Run the full experiment
    perform_experiment(
        X=X,
        y=y,
        instance_index=42,
        name="full_qnn_experiment",
        k_external=5,
        k_internal=3,
        scoring="accuracy",
        path="results_qnn/",
        scale_data=True,
        n_jobs=1
    )
