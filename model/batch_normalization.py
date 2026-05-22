import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        
        mean_batch = np.mean(x, axis=0)
        var_batch = np.var(x, axis=0)

        running_mean = np.array(running_mean)
        running_var = np.array(running_var)

        x_hat = (x - mean_batch) / (np.sqrt(var_batch + eps))
        y = gamma * x_hat + beta
        if training:
            running_mean = (1 - momentum) * running_mean + momentum * mean_batch
            running_var = (1 - momentum) * running_var + momentum * var_batch
        else:
            x_hat = (x - running_mean) / np.sqrt(running_var + eps)
            y = gamma * x_hat + beta
            return tuple((np.round(y,4), np.round(running_mean,4), np.round(running_var,4)))
        return tuple((np.round(y,4), np.round(running_mean,4), np.round(running_var,4)))

