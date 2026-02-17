# test_qnn_minimal.py
import pennylane as qml
from pennylane import numpy as np
import sys

print("1. Imports OK")

# Copy just the PennyQNN class here (the improved version)
# ... (I'll provide it)

print("2. Class defined")

# Test with tiny dataset
X_test = np.random.random((10, 4))
y_test = np.array([0,1,0,1,0,1,0,1,0,1])

qnn = PennyQNN(n_qubits=2, epochs=2)
print("3. QNN initialized")

qnn.fit(X_test, y_test)
print("4. Training completed")

preds = qnn.predict(X_test)
print(f"5. Predictions: {preds}")
print(f"6. Accuracy: {qnn.score(X_test, y_test)}")
