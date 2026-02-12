import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from contextlib import contextmanager
from spikingjelly.activation_based import layer, functional, neuron

def normalize_per_timestep(cams: torch.Tensor, eps: float = 1e-6, q: float = 99.0) -> torch.Tensor:

    B, T, H, W = cams.shape
    flat = cams.view(B, T, -1)

    # lower/upper percentiles per (B,T)
    lo = torch.quantile(flat, (100.0 - q) / 100.0, dim=-1, keepdim=True)
    hi = torch.quantile(flat, q / 100.0, dim=-1, keepdim=True)
    lo = lo.view(B, T, 1, 1)
    hi = hi.view(B, T, 1, 1)

    cams = (cams - lo) / (hi - lo + eps)

    return cams.clamp(0.0, 1.0)

def upsample_cams(cams: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    # Accept (B,T,H,W) or (T,H,W)
    squeeze = False
    if cams.dim() == 3:
        cams = cams.unsqueeze(0)
        squeeze = True
    B, T, Hf, Wf = cams.shape
    cams_b1 = cams.view(B * T, 1, Hf, Wf)
    cams_up = F.interpolate(cams_b1, size=target_hw, mode="bilinear", align_corners=False)
    cams_up = cams_up.view(B, T, *target_hw)
    if squeeze:
        cams_up = cams_up.squeeze(0)
    return cams_up

@contextmanager
def grad_cam_context(model: nn.Module):
    """
    Context manager to temporarily enable gradients and set training mode 
    (preserving BN/Dropout eval state) for GradCAM.
    """
    prev_training = model.training   

    prev_grad_states = {p: p.requires_grad for p in model.parameters()}
    # 1. Enable Grads
    for p in model.parameters():
        p.requires_grad = True 
    
    # 2. Set Global Eval mode
    # This ensures all BN (including ThresholdDependentBatchNorm), Dropout, etc. are frozen.
    # This is safer than trying to blacklist every possible normalization layer.
    model.eval()

    # 3. Selectively set Neurons to Train mode
    # Only necessary if surrogate gradients or neuron dynamics specifically require it.
    for m in model.modules():
        if isinstance(m, neuron.BaseNode):
            m.train()

    try:
        yield
    finally:
        # 4. Cleanup: Zero grads, disable grads, restore mode, reset state
        model.zero_grad()
        for p in model.parameters():
            for p, req_grad in prev_grad_states.items():
                p.requires_grad = req_grad
        model.train(prev_training)
        functional.reset_net(model)

def mask_border_pixels(cams: torch.Tensor, border_widths: Tuple[int, int] = (1, 1)) -> torch.Tensor:
    """
    Zeroes out the spatial borders of the CAM to remove padding artifacts.
    Assumes cams shape is (..., H, W).
    border_widths: (pad_h, pad_w) matching PyTorch Conv2d padding tuple format.
    """
    pad_h, pad_w = border_widths
    
    if pad_h <= 0 and pad_w <= 0:
        return cams
    
    H, W = cams.shape[-2], cams.shape[-1]
    
    # Mask Height (Top/Bottom)
    if pad_h > 0 and H > 2 * pad_h:
        cams[..., :pad_h, :] = 0
        cams[..., -pad_h:, :] = 0
            
    # Mask Width (Left/Right)
    if pad_w > 0 and W > 2 * pad_w:
        cams[..., :, :pad_w] = 0
        cams[..., :, -pad_w:] = 0
    return cams