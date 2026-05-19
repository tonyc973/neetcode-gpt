import numpy as np
from typing import List

class Solution:
    def forward_and_backward(self,
                             x: List[float],
                             W1: List[List[float]], b1: List[float],
                             W2: List[List[float]], b2: List[float],
                             y_true: List[float]) -> dict:
        
        # Convert to numpy arrays
        x = np.array(x)
        W1 = np.array(W1)
        b1 = np.array(b1)
        W2 = np.array(W2)
        b2 = np.array(b2)
        y_true = np.array(y_true)
        
        # --- FORWARD PASS ---
        z1 = np.dot(W1, x) + b1
        a1 = np.maximum(z1, 0)
        z2 = np.dot(W2, a1) + b2
        predictions = z2
        
        loss = np.mean((predictions - y_true)**2)
        
        # --- BACKWARD PASS ---
        # Derivative of MSE: 2/N * (pred - y)
        dz2 = (2.0 / len(y_true)) * (predictions - y_true)
        
        # dW2 = dz2 * a1^T
        dW2 = np.outer(dz2, a1)
        db2 = dz2
        
        # Gradient w.r.t a1: W2^T * dz2
        da1 = np.dot(W2.T, dz2)
        
        # Gradient w.r.t z1 (ReLU derivative)
        dz1 = da1 * (z1 > 0)
        
        # dW1 = dz1 * x^T
        dW1 = np.outer(dz1, x)
        db1 = dz1
        
        # Helper to ensure 0.0 instead of -0.0
        def clean(arr):
            arr[np.abs(arr) < 1e-9] = 0.0
            return np.round(arr, 4).tolist()

        return {
            'loss': round(float(loss), 4),
            'dW1': clean(dW1),
            'db1': clean(db1),
            'dW2': clean(dW2),
            'db2': clean(db2)
        }