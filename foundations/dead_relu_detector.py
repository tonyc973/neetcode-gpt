import torch
import torch.nn as nn
from typing import List


class Solution:
    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        dead_fractions = []
        out = x
        for layer in model:
            out = layer(out)                      # push the data THROUGH this layer
            if isinstance(layer, nn.ReLU):
                # out shape: (batch_size, num_neurons)
                dead_per_neuron = (out == 0).all(dim=0)   # True for each all-zero column
                fraction = dead_per_neuron.float().mean().item()
                dead_fractions.append(round(fraction, 4))
        return dead_fractions



    def suggest_fix(self, dead_fractions: List[float]) -> str:
        # 1. any layer mostly dead -> the activation itself is the problem
        if any(f > 0.5 for f in dead_fractions):
            return 'use_leaky_relu'

        # 2. first layer too dead -> bad starting weights
        if dead_fractions[0] > 0.3:
            return 'reinitialize'

        # 3. deadness grows layer by layer AND the last layer is affected -> training too aggressive
        strictly_increasing = all(
            dead_fractions[i] < dead_fractions[i + 1]
            for i in range(len(dead_fractions) - 1)
        )
        if strictly_increasing and dead_fractions[-1] > 0.1:
            return 'reduce_learning_rate'

        # 4 & 5. nothing alarming
        return 'healthy'
