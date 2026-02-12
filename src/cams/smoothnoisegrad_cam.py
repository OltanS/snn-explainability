import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from spikingjelly.activation_based import functional
from src.cams.base_temporal_cam import BaseTemporalCAM
from src.noisy_node import noise_injection
from src.cams.cam_utils import mask_border_pixels

class SNNSmoothNoiseGradCAM(BaseTemporalCAM):
    """
    Doubly Stochastic GradCAM:
    Combines SmoothGrad (Input Noise) and NoiseGradCAM (Internal/Surrogate Noise).
    
    Instead of nesting loops (Cost: N_smooth * N_internal), we run a single loop
    of 'k_trials' where EACH trial has:
      1. Unique random noise added to the input (Data Robustness).
      2. Unique random noise injected into network thresholds (Model Robustness).
    """
    def __init__(self, 
                 model: nn.Module, 
                 target_layer: nn.Module, 
                 k_trials: int = 100, 
                 batch_size: int = 16, 
                 stdev_spread: float = 0.15,
                 noise_type: str = "uniform"):
        super().__init__(model, target_layer)
        self.k_trials = k_trials
        self.batch_size = batch_size
        self.stdev_spread = stdev_spread
        self.noise_type = noise_type
        self.mask_borders = False


    def _compute_raw(self, x: torch.Tensor, class_idx: Optional[int]) -> torch.Tensor:
        # x: (T, B, C, H, W)
        T, B, C_in, H, W = x.shape

        # Calculate noise scale for SmoothGrad (Input Noise)
        # Based on (max - min) of the input.
        val_range = (x.max() - x.min()).item()
        if val_range == 0: 
            val_range = 1.0
        sigma_input = self.stdev_spread * val_range

        # Determine target classes
        if class_idx is None:
            with torch.no_grad():
                functional.reset_net(self.model)
                out_clean = self.model(x)
                target_classes = out_clean.mean(dim=0).argmax(dim=1)  # (B,)
                functional.reset_net(self.model)
        else:
            target_classes = torch.full((B,), int(class_idx), device=x.device, dtype=torch.long)
        
        sum_cams = None

        # Activate Internal Noise Context (The "NoiseGrad" part)
        # This injects noise into thresholds/membranes downstream of target_layer
        with noise_injection(self.model, noise_type=self.noise_type, target_layer=self.target_layer):
            trials_run = 0
            
            while trials_run < self.k_trials:
                current_batch_k = min(self.batch_size, self.k_trials - trials_run)
                
                # 1. Prepare Batch: Repeat x
                # (T, B, C, H, W) -> (T, B*k, C, H, W)
                x_rep = x.repeat(1, current_batch_k, 1, 1, 1)
                
                # 2. Apply Input Noise (The "SmoothGrad" part)
                # We generate unique noise for every sample in the repeated batch
                input_noise = torch.randn_like(x_rep) * sigma_input
                x_noisy = x_rep + input_noise
                
                x_noisy = x_noisy.detach().requires_grad_(True)
                
                acts = None
                grads = None
                
                # Hook for activations and gradients
                def fwd_hook(module, inp, out):
                    nonlocal acts
                    acts = out
                    out.retain_grad()
                    def grad_hook(g):
                        nonlocal grads
                        grads = g
                    out.register_hook(grad_hook)

                handle = self.target_layer.register_forward_hook(fwd_hook)
                
                try:
                    self.model.zero_grad(set_to_none=True)
                    functional.reset_net(self.model)
                    
                    with torch.enable_grad():
                        # Forward pass with BOTH input noise (x_noisy) AND internal noise (context manager)
                        outputs = self.model(x_noisy)  # (T, B*k, K)
                        
                        target_classes_rep = target_classes.repeat(current_batch_k)
                        T_out, Bk, K = outputs.shape
                        
                        # Gather scores for target class
                        # We use mean over time for the score
                        target_score = outputs.gather(2, target_classes_rep.view(1, Bk, 1).expand(T_out, Bk, 1)).mean()
                        target_score.backward()
                    
                    if acts is None or grads is None:
                        raise RuntimeError("Hooks failed to capture activations/gradients. Check target_layer validity.")

                    # Compute GradCAM weights: Global Average Pooling of Gradients
                    # grads: (T, Bk, C, H, W) -> w: (T, Bk, C)
                    w = grads.mean(dim=(-2, -1))
                    
                    # Reshape for broadcast: (T, Bk, C, 1, 1)
                    w = w.unsqueeze(-1).unsqueeze(-1)
                    
                    # Weighted combination: w * acts
                    # (T, Bk, C, 1, 1) * (T, Bk, C, H, W) -> (T, Bk, C, H, W)
                    cam_weighted = w * acts
                    
                    # Sum over channels -> (T, Bk, H, W)
                    batch_cams = cam_weighted.sum(dim=2)
                    
                    # Reshape to separate trials: (T, k, B, H, W)
                    batch_cams = batch_cams.view(T_out, current_batch_k, B, batch_cams.shape[-2], batch_cams.shape[-1])
                    
                    # Sum over trials (k) to accumulate
                    batch_cams_sum = batch_cams.sum(dim=1) # (T, B, H, W)
                    
                    if sum_cams is None:
                        sum_cams = batch_cams_sum.detach()
                    else:
                        sum_cams += batch_cams_sum.detach()
                        
                finally:
                    handle.remove()
                    functional.reset_net(self.model)
                    acts = None
                    grads = None
                
                trials_run += current_batch_k

        # Average and ReLU
        avg_cams = sum_cams / self.k_trials
        avg_cams = F.relu(avg_cams)
        if self.mask_borders:
            avg_cams = mask_border_pixels(avg_cams, border_widths=self.target_layer.padding)
        # Permute to standard format: (B, T, Hf, Wf)
        avg_cams = avg_cams.permute(1, 0, 2, 3)
        return avg_cams