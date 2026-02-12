import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from spikingjelly.activation_based import functional
from src.cams.base_temporal_cam import BaseTemporalCAM

class SNNGrad3D(BaseTemporalCAM):
    """
    Compute gradients of the class score (sum of output spikes per output neuron)
    w.r.t. the input tensor to produce a spatio-temporal saliency map.
    Input x expected shape: (T, B, C, H, W)
    Returns raw cams at input resolution: (B, T, H, W)
    The returned gradients are signed (can be positive or negative). Post-processing
    (abs, ReLU, normalization) can be done by caller / BaseTemporalCAM.generate.
    """

    def __init__(self, model: nn.Module):
        # target_layer is unused for this CAM, but BaseTemporalCAM requires it.
        super().__init__(model, target_layer=None)

    def _compute_raw(self, x: torch.Tensor, class_idx: Optional[int]) -> torch.Tensor:
        # x: (T, B, C, H, W)
        assert x.dim() == 5
        device = next(self.model.parameters()).device
        x = x.to(device).detach().requires_grad_(True)

        functional.reset_net(self.model)
        self.model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            outputs = self.model(x)  # (T, B, K)

            # Class score: sum of spikes across time -> (B, K)
            scores = outputs.sum(dim=0)

            B = scores.shape[0]
            if class_idx is None:
                cls = scores.argmax(dim=1)  # (B,)
            else:
                cls = torch.full((B,), int(class_idx), device=scores.device, dtype=torch.long)

            # We'll compute per-sample gradients (scalar per sample)
            cams_per_batch = []
            for b in range(B):
                target_score = scores[b, cls[b]]
                # Compute gradient of scalar target_score w.r.t input x
                grad_x = torch.autograd.grad(target_score, x, retain_graph=True, create_graph=False)[0]  # (T,B,C,H,W)
                # extract entries for this sample
                grad_b = grad_x[:, b, :, :, :]  # (T, C, H, W)
                # collapse channels -> (T, H, W). Use signed sum over channels (preserving sign)
                cam_b = grad_b.sum(dim=1)  # (T, H, W)
                cams_per_batch.append(cam_b)

            cams = torch.stack(cams_per_batch, dim=0)  # (B, T, H, W)

        functional.reset_net(self.model)
        return cams