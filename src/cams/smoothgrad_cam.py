import torch
import torch.nn as nn
from typing import Optional
from src.cams.base_temporal_cam import BaseTemporalCAM
from src.cams.gradcam import SNNGradCAM

class SNNSmoothGradCAM(BaseTemporalCAM):
    """
    SmoothGrad-CAM for SNNs.
    Adds Gaussian noise to the input events/tensor multiple times,
    computes Grad-CAM for each noisy input, and averages the resulting CAMs.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module, n_samples: int = 25, stdev_spread: float = 0.15):
        super().__init__(model, target_layer)
        self.n_samples = n_samples
        self.stdev_spread = stdev_spread
        self.grad_cam = SNNGradCAM(model, target_layer)

    def _compute_raw(self, x: torch.Tensor, class_idx: Optional[int]) -> torch.Tensor:
        # x: (T, B, C, H, W)
        
        # Calculate standard deviation for noise
        # SmoothGrad suggests stdev = stdev_spread * (max - min)
        # For binary events, max-min is usually 1.
        sigma = self.stdev_spread * (x.max() - x.min()).item()
        
        sum_cams = None
        
        for i in range(self.n_samples):
            noise = torch.randn_like(x) * sigma
            noisy_x = x + noise
            
            # Compute CAM for noisy input
            # SNNGradCAM._compute_raw returns (B, T, Hf, Wf)
            # It handles hook registration and network reset internally.
            cams = self.grad_cam._compute_raw(noisy_x, class_idx)
            
            if sum_cams is None:
                sum_cams = cams
            else:
                sum_cams += cams
        
        avg_cams = sum_cams / self.n_samples
        return avg_cams