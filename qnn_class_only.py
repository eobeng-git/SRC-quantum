"""
Improved PennyQNN Class - Test Version
"""

import pennylane as qml
from pennylane import numpy as np
import sys

print("✓ Imports successful")

class PennyQNN:
    """
    Improved PennyLane QNN classifier with proper training
    """
    def __init__(self, n_qubits=2, n_layers=2, epochs=5, learning_rate=0.1, batch_size=4):
        """
        Initialize the QNN classifier
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        print(f"  Initializing QNN with {n_qubits} qubits, {n_layers} layers")
        sys.stdout.flush()
        
        # Initialize weights randomly
        weight_shape = (n_layers, n_qubits, 2)
        self.weights = np.random.random(weight_shape, requires_grad=True)
        
        # Initialize device and circuit
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.circuit = self._create_circuit()
        
        self.trained = False
        self.X_min = None
        self.X_max = None
        
        print("  ✓ QNN initialized")
        sys.stdout.flush()
    
    def _create_circuit(self):
        """Create the quantum circuit"""
        
        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs, weights):
            # Encode classical data
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits), rotation='Y')
            
            # Variational layers
            for layer in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)
                
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            
            # Measure expectation value
            return qml.expval(qml.PauliZ(0))
        
        return circuit
    
    def _preprocess_features(self, X):
        """Simple preprocessing - just ensure correct dimension"""
        X = np.array(X)
        
        # Handle 1D array
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Simple normalization to [0, 1]
        X_min = X.min(axis=0, keepdims=True)
        X_max = X.max(axis=0, keepdims=True)
        X_normalized = (X - X_min) / (X_max - X_min + 1e-8)
        
        # Handle feature dimension
        n_features = X.shape[1]
        X_processed = []
        
        for x in X_normalized:
            if n_features > self.n_qubits:
                # Take first n_qubits features (simple for now)
                X_processed.append(x[:self.n_qubits])
            elif n_features < self.n_qubits:
                # Pad with zeros
                padded = np.zeros(self.n_qubits)
                padded[:n_features] = x
                X_processed.append(padded)
            else:
                X_processed.append(x)
        
        return np.array(X_processed)
    
    def fit(self, X, y):
        """Train the QNN"""
        print(f"\nStarting QNN Training...")
        sys.stdout.flush()
        
        # Preprocess
        X_processed = self._preprocess_features(X)
        y = np.array(y)
        
        # Simple training loop
        opt = qml.GradientDescentOptimizer(stepsize=self.learning_rate)
        
        for epoch in range(self.epochs):
            # Define cost function
            def cost(weights):
                predictions = []
                for x in X_processed:
                    val = self.circuit(x, weights)
                    predictions.append(val)
                predictions = np.array(predictions)
                return np.mean((predictions - y) ** 2)
            
            # Update weights
            self.weights = opt.step(cost, self.weights)
            
            # Calculate accuracy
            train_acc = self.score(X, y)
            print(f"  Epoch {epoch+1}/{self.epochs} | Accuracy: {train_acc:.4f}")
            sys.stdout.flush()
        
        self.trained = True
        print("  ✓ Training completed")
        sys.stdout.flush()
        return self
    
    def predict(self, X):
        """Predict labels"""
        X_processed = self._preprocess_features(X)
        
        predictions = []
        for x in X_processed:
            val = float(self.circuit(x, self.weights))
            predictions.append(1 if val > 0 else 0)
        
        return np.array(predictions)
    
    def score(self, X, y):
        """Calculate accuracy"""
        preds = self.predict(X)
        return np.mean(preds == y)

print("✓ Class defined successfully")

# ============================================
# TEST THE CLASS WITH DUMMY DATA
# ============================================
print("\n" + "="*50)
print("TESTING QNN CLASS")
print("="*50)

# Create dummy data
X_dummy = np.random.random((20, 4))
y_dummy = np.array([0,1]*10)  # Alternating 0,1

print(f"Created dummy data: X shape {X_dummy.shape}, y shape {y_dummy.shape}")

# Create QNN instance
try:
    qnn = PennyQNN(n_qubits=2, epochs=3)
    print("✓ QNN instance created")
except Exception as e:
    print(f"✗ Failed to create QNN: {e}")

# Test training
try:
    qnn.fit(X_dummy, y_dummy)
    print("✓ Training completed")
except Exception as e:
    print(f"✗ Training failed: {e}")

# Test prediction
try:
    preds = qnn.predict(X_dummy[:5])
    print(f"✓ Predictions: {preds}")
except Exception as e:
    print(f"✗ Prediction failed: {e}")

# Test score
try:
    acc = qnn.score(X_dummy, y_dummy)
    print(f"✓ Accuracy: {acc:.4f}")
except Exception as e:
    print(f"✗ Score failed: {e}")

print("\n" + "="*50)
print("TEST COMPLETE")
print("="*50)
