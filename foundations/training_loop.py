import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        n = X.shape[0]
        w = np.zeros(X.shape[1])   # initialize once, before the loop
        b = 0.0

        for _ in range(epochs):
            y_hat = np.dot(X, w) + b
            error = y_hat - y
            gradient_w = (2 / n) * np.dot(X.T, error)
            gradient_b = (2 / n) * np.sum(error)
            w = w - lr * gradient_w
            b = b - lr * gradient_b

        return (np.round(w, 5), round(b, 5))