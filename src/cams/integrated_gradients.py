import torch
import torch.nn as nn
from typing import Optional

from spikingjelly.activation_based import functional
from src.cams.base_temporal_cam import BaseTemporalCAM
from src.cams.grad3d import SNNGrad3D

class SNNIntegratedGradients(BaseTemporalCAM):
    """
    Integrated Gradients adapted for spiking inputs.

    mode:
      - '3d' : path along the time dimension (prefix-based sampling), baseline defaults to zeros (no spikes).
      - '2d' : straight-line interpolation between baseline and input in the input-value space
               (useful for encoded images / continuous-valued inputs).

    steps: number of integration steps (Riemann sum).
    """

    def __init__(self, model: nn.Module, steps: int = 50, mode: str = "3d", baseline: Optional[torch.Tensor] = None):
        # target_layer not needed; use BaseTemporalCAM signature and SNNGrad3D to compute gradients.
        super().__init__(model, target_layer=None)
        self.steps = int(steps)
        assert mode in ("3d", "2d"), "mode must be '3d' or '2d'"
        self.mode = mode
        self.baseline = baseline
        self._grad3d = SNNGrad3D(model)

    def _compute_raw(self, x: torch.Tensor, class_idx: Optional[int]) -> torch.Tensor:
        # x: (T, B, C, H, W)
        assert x.dim() == 5
        device = next(self.model.parameters()).device
        x = x.to(device)

        T, B, C, H, W = x.shape

        # baseline
        if self.baseline is None:
            baseline = torch.zeros_like(x)
        else:
            baseline = self.baseline.to(device)
            assert baseline.shape == x.shape

        grads_sum = None

        if self.mode == "2d":
            # Straight-line interpolation in input-space
            for s in range(1, self.steps + 1):
                alpha = float(s) / self.steps
                x_step = baseline + alpha * (x - baseline)
                # compute signed gradients at this step
                g = self._grad3d._compute_raw(x_step.detach(), class_idx)  # (B, T, H, W)
                if grads_sum is None:
                    grads_sum = g
                else:
                    grads_sum = grads_sum + g
        else:
            # mode == '3d': build prefix-based path where we include progressively more timesteps
            for s in range(1, self.steps + 1):
                # include first k timesteps
                k = max(1, int(round(T * s / self.steps)))
                x_step = torch.zeros_like(x)
                x_step[:k] = x[:k]
                g = self._grad3d._compute_raw(x_step.detach(), class_idx)  # (B, T, H, W)
                if grads_sum is None:
                    grads_sum = g
                else:
                    grads_sum = grads_sum + g

        avg_grads = grads_sum / float(self.steps)  # (B, T, H, W)

        # Integrated gradients: (input - baseline) * average_grads
        delta = (x - baseline)  # (T,B,C,H,W)
        # collapse channels of delta to match avg_grads shape:
        # delta_ch_sum: (T, B, H, W) -> permute to (B, T, H, W)
        delta_ch_sum = delta.sum(dim=2).permute(1, 0, 2, 3)
        attributions = delta_ch_sum * avg_grads  # (B, T, H, W)

        # Return attributions (signed). Caller may apply abs/ReLU/normalize.
        return attributions