import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List
from contextlib import contextmanager

from spikingjelly.activation_based import functional, layer, surrogate
from src.cams.base_temporal_cam import BaseTemporalCAM

class ReplaceSurrogateGradCAM(BaseTemporalCAM):
    def __init__(self, model: nn.Module, target_layer: nn.Module, new_surrogate = None):
        super().__init__(model, target_layer)
        # The requested surrogate is a shifted ReLU where gradient is 1 if v > v_th else 0.
        # LeakyKReLU with leak=0 and k=1 achieves this:
        # Forward: Heaviside(x)
        # Backward: 1 if x >= 0 else 0
        if new_surrogate is None:
            self.grad_surrogate = surrogate.LeakyKReLU(leak=0.0, k=1.0)
        else:
            self.grad_surrogate = new_surrogate

    @contextmanager
    def _override_surrogate(self):
        """
        Context manager to temporarily replace the surrogate function of all 
        spiking neurons in the model with self.grad_surrogate.
        """
        original_surrogates = {}
        # Identify modules that have a surrogate_function attribute (e.g. LIFNode, IFNode)
        modules_with_surrogate = [m for m in self.model.modules() if hasattr(m, 'surrogate_function')]
        
        # Swap
        for module in modules_with_surrogate:
            original_surrogates[module] = module.surrogate_function
            module.surrogate_function = self.grad_surrogate
        
        try:
            yield
        finally:
            # Restore
            for module, orig_surr in original_surrogates.items():
                module.surrogate_function = orig_surr

    def _compute_raw(self, x: torch.Tensor, class_idx: Optional[int]) -> torch.Tensor:
        acts = None
        grads = None
        
        def fwd_hook(module, inp, out):
            nonlocal acts
            # out: (T,B,C,H,W)
            acts = out
            out.retain_grad()
            def grad_hook(g):
                nonlocal grads
                grads = g
            out.register_hook(grad_hook)

        handle = self.target_layer.register_forward_hook(fwd_hook)

        x = x.detach().requires_grad_(True)
        self.model.zero_grad(set_to_none=True)
        functional.reset_net(self.model)
        
        try:
            with self._override_surrogate():
                with torch.enable_grad():
                    outputs = self.model(x)  # (T,B,K)

                    # Activations should be captured during forward
                    assert acts is not None, "Missing activations."
                    
                    T, B, C, Hf, Wf = acts.shape

                    if class_idx is None:
                        cls = outputs.mean(dim=0).argmax(dim=1)  # (B,)
                    else:
                        cls = torch.full((B,), int(class_idx), device=outputs.device, dtype=torch.long)

                    target = outputs.gather(2, cls.view(1, B, 1).expand(T, B, 1)).mean()
                    target.backward()  # this triggers the grad hook

                # Gradients are available only after backward
                assert grads is not None, "Missing gradients (ensure target layer output requires grad)."
                
                weights = grads.mean(dim=(-2, -1))                              # (T,B,C)
                cams = (weights.unsqueeze(-1).unsqueeze(-1) * acts).sum(dim=2)  # (T,B,Hf,Wf)
                cams = cams.permute(1, 0, 2, 3)                                 # (B,T,Hf,Wf)
                cams = F.relu(cams)
                
        finally:
            handle.remove()
            # Break references to graph
            acts = None
            grads = None
            
        return cams