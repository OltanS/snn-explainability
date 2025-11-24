import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List

from spikingjelly.activation_based import functional, layer
from src.cams.base_temporal_cam import BaseTemporalCAM

class SNNGradCAM(BaseTemporalCAM):
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        super().__init__(model, target_layer)
        self._acts = None
        self._grads = None
        self._handle = self.target_layer.register_forward_hook(self._fwd_hook)

    def _fwd_hook(self, module, inp, out):
        # out: (T,B,C,H,W) or (B,C,H,W)? For multi-step aggregated modules in SpikingJelly often (T,B,C,H,W).
        self._acts = out
        out.retain_grad()
        def grad_hook(g):
            self._grads = g
        out.register_hook(grad_hook)

    def remove_hooks(self):
        if self._handle:
            self._handle.remove()
            self._handle = None

    def _compute_raw(self, x: torch.Tensor, class_idx: Optional[int]) -> torch.Tensor:
        self._acts = None
        self._grads = None
        x = x.detach().requires_grad_(True)
        self.model.zero_grad(set_to_none=True)
        functional.reset_net(self.model)
        with torch.enable_grad():
            outputs = self.model(x)  # (T,B,K)

        # Activations should be captured during forward
        assert self._acts is not None, "Missing activations."
        acts = self._acts  # (T,B,C,Hf,Wf)
        T, B, C, Hf, Wf = acts.shape

        if class_idx is None:
            cls = outputs.mean(dim=0).argmax(dim=1)  # (B,)
        else:
            cls = torch.full((B,), int(class_idx), device=outputs.device, dtype=torch.long)

        target = outputs.gather(2, cls.view(1, B, 1).expand(T, B, 1)).mean()
        target.backward()  # this triggers the grad hook

        # Gradients are available only after backward
        assert self._grads is not None, "Missing gradients (ensure target layer output requires grad)."
        grads = self._grads

        weights = grads.mean(dim=(-2, -1))                              # (T,B,C)
        cams = (weights.unsqueeze(-1).unsqueeze(-1) * acts).sum(dim=2)  # (T,B,Hf,Wf)
        cams = cams.permute(1, 0, 2, 3)                                 # (B,T,Hf,Wf)
        cams = F.relu(cams)
        return cams