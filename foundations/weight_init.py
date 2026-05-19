import torch
import torch.nn as nn
import numpy as np
from typing import List

class Solution:
    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Xavier/Glorot normal: std = sqrt(2 / (fan_in + fan_out))
        torch.manual_seed(0)
        std = np.sqrt(2.0 / (fan_in + fan_out))
        weights = torch.randn(fan_out, fan_in) * std
        return np.round(weights.numpy(), 4).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Kaiming/He normal for ReLU: std = sqrt(2 / fan_in)
        torch.manual_seed(0)
        std = np.sqrt(2.0 / fan_in)
        weights = torch.randn(fan_out, fan_in) * std
        return np.round(weights.numpy(), 4).tolist()

    def check_activations(self, num_layers, input_dim, hidden_dim, init_type):
        torch.manual_seed(0)
        
        # Build all layers first, then forward pass — standard PyTorch pattern
        layers = []
        for i in range(num_layers):
            f_in = input_dim if i == 0 else hidden_dim
            f_out = hidden_dim
            
            if init_type == 'xavier':
                std = np.sqrt(2.0 / (f_in + f_out))
            elif init_type == 'kaiming':
                std = np.sqrt(2.0 / f_in)
            else:
                std = 1.0
            
            W = torch.randn(f_out, f_in) * std
            layers.append(W)
        
        # Generate input AFTER weights
        x = torch.randn(input_dim)
        
        stds = []
        for W in layers:
            x = torch.relu(W @ x)
            stds.append(round(x.std().item(), 2))
        
        return stds