import torch
import torch.nn.functional as F
from typing import Tuple

def normalize_per_timestep(cams: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # Accept (B,T,H,W) or (T,H,W)
    squeeze = False
    if cams.dim() == 3:
        cams = cams.unsqueeze(0)
        squeeze = True
    B, T, H, W = cams.shape
    flat = cams.view(B, T, -1)
    mins = flat.min(dim=-1).values.view(B, T, 1, 1)
    maxs = flat.max(dim=-1).values.view(B, T, 1, 1)
    denom = (maxs - mins).clamp_min(eps)
    out = ((cams - mins) / denom).clamp(0, 1)
    if squeeze:
        out = out.squeeze(0)
    return out

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