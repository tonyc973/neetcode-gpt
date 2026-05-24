import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        stats = []
        with torch.no_grad():
            out = x
            for layer in model:
                out = layer(out)
                if isinstance(layer,nn.Linear):
                    dead_per_neuron = (out <= 0).all(dim = 0)
                    dead_fraction = dead_per_neuron.float().mean().item()
                    d = {
                        "mean" : round(out.mean().item(), 4),
                        "std" : round(out.std().item(), 4),
                        "dead_fraction": round(dead_fraction, 4)
                    }
                    stats.append(d)
        return stats




    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad()
        out = model(x)
        loss = nn.MSELoss()(out, y)
        loss.backward()

        stats = []
        for layer in model:
            if isinstance(layer, nn.Linear):
                w = layer.weight.grad
                stats.append({
                    "mean" : round(w.mean().item(), 4),
                    "std" : round(w.std().item(), 4),
                    "norm" : round(torch.norm(w).item(), 4)
                })
        return stats


    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        if any(s['dead_fraction'] > 0.5 for s in activation_stats):
            return 'dead_neurons'
        if any(s['norm'] > 1000 for s in gradient_stats):
            return 'exploding_gradients'

        # 3. vanishing gradients (last layer norm < 1e-5)
        if gradient_stats and gradient_stats[-1]['norm'] < 1e-5:
            return 'vanishing_gradients'

        # 4. activation std checks
        for s in activation_stats:
            if s['std'] < 0.1:
                return 'vanishing_gradients'
            if s['std'] > 10.0:
                return 'exploding_gradients'

        # 5. healthy
        return 'healthy'

        
