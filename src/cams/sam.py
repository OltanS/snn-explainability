import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from spikingjelly.activation_based import functional
from src.cams.base_temporal_cam import BaseTemporalCAM

class SpikeActivationMap(BaseTemporalCAM):
    def __init__(self, model: nn.Module, target_layer: nn.Module, gamma: float = 0.5):
        super().__init__(model, target_layer)
        self.gamma = gamma
    
    def _compute_raw(self, x: torch.Tensor, class_idx: Optional[int]) -> torch.Tensor:
        seq = None
        def hook(module, inp, out):
            nonlocal seq
            # Expect multi-step aggregated output: (T,B,C,H,W)
            seq = out.detach()
            
        handle = self.target_layer.register_forward_hook(hook)
        
        try:
            _ = self.model(x)  # Forward to collect spikes.
        finally:
            handle.remove()
            
        assert seq is not None, "No spike sequence captured."
        spk_seq = seq  # (T,B,C,Hf,Wf)
        T, B, C, Hf, Wf = spk_seq.shape
        E = torch.exp(torch.tensor(-self.gamma, device=spk_seq.device, dtype=spk_seq.dtype))
        cams = torch.zeros(B, T, Hf, Wf, device=spk_seq.device)
        N_prev = torch.zeros(B, C, Hf, Wf, device=spk_seq.device)
        for t in range(T):
            S_t = spk_seq[t]              # (B,C,Hf,Wf)
            cams[:, t] = (N_prev * S_t).sum(dim=1)
            N_prev = E * (N_prev + S_t)
        return cams