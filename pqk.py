"""
Code for experiments from the paper:
`Potential of Quantum Machine Learning for Processing Multispectral Earth Observation Data'
This version has been anonymized for the purpose of peer review and may not be used for any other purpose. 
The code for the experiments will be publicly available under an open license - links to the repository 
will be found in the paper.
-------------------------------
PQK implementation
"""

import os
import cirq
import sympy
import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from joblib import Memory


###############################################
# Utilitary functions to build quantum circuit
###############################################
def single_qubit_wall(qubits, rotations):
    """Prepare a single qubit X,Y,Z rotation wall on `qubits`."""
    wall_circuit = cirq.Circuit()
    for i, qubit in enumerate(qubits):
        for j, gate in enumerate([cirq.X, cirq.Y, cirq.Z]):
            wall_circuit.append(gate(qubit) ** rotations[i][j])

    return wall_circuit


def v_theta(qubits):
    """Prepares a circuit that generates V(\theta)."""
    ref_paulis = [
        cirq.X(q0) * cirq.X(q1) + cirq.Y(q0) * cirq.Y(q1) + cirq.Z(q0) * cirq.Z(q1)
        for q0, q1 in zip(qubits, qubits[1:])
    ]
    exp_symbols = list(sympy.symbols("ref_0:" + str(len(ref_paulis))))
    return tfq.util.exponential(ref_paulis, exp_symbols), exp_symbols


def prepare_pqk_circuits(qubits, classical_source, random_rots, n_trotter=10):
    """Prepare the pqk feature circuits around a dataset."""
    n_qubits = len(qubits)
    n_points = len(classical_source)

    # Prepare random single qubit rotation wall.
    initial_U = single_qubit_wall(qubits, random_rots)

    # Prepare parametrized V
    V_circuit, symbols = v_theta(qubits)
    exp_circuit = cirq.Circuit(V_circuit for t in range(n_trotter))

    # Convert to `tf.Tensor`
    initial_U_tensor = tfq.convert_to_tensor([initial_U])
    initial_U_splat = tf.tile(initial_U_tensor, [n_points])

    full_circuits = tfq.layers.AddCircuit()(initial_U_splat, append=exp_circuit)

    # Replace placeholders in circuits with values from `classical_source`.
    return tfq.resolve_parameters(
        full_circuits,
        tf.convert_to_tensor([str(x) for x in symbols]),
        tf.convert_to_tensor(classical_source * (n_qubits / 3) / n_trotter),
    )


def get_pqk_features(qubits, data_batch):
    """Get PQK features based on above construction."""
    ops = [[cirq.X(q), cirq.Y(q), cirq.Z(q)] for q in qubits]
    ops_tensor = tf.expand_dims(tf.reshape(tfq.convert_to_tensor(ops), -1), 0)
    batch_dim = tf.gather(tf.shape(data_batch), 0)
    ops_splat = tf.tile(ops_tensor, [batch_dim, 1])
    exp_vals = tfq.layers.Expectation()(data_batch, operators=ops_splat)
    rdm = tf.reshape(exp_vals, [batch_dim, len(qubits), -1])
    return rdm


class PqkTransform(BaseEstimator, TransformerMixin):
    def __init__(self, random_rots, n_trotter=10):
        self.n_trotter = n_trotter
        self.random_rots = random_rots

    def fit(self, X, y=None):
        self.dataset_dim_ = X.shape[1]
        return self

    def transform(self, X):
        # make sure that it was fitted
        check_is_fitted(self, "dataset_dim_")

        qubits = cirq.GridQubit.rect(1, self.dataset_dim_ + 1)
        q_x_circuits = prepare_pqk_circuits(qubits, X, self.random_rots, self.n_trotter)
        x_pqk = get_pqk_features(qubits, q_x_circuits)

        return tf.reshape(x_pqk, [X.shape[0], -1]).numpy()
