import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List

from spikingjelly.activation_based import functional, neuron
from src.cams.base_temporal_cam import BaseTemporalCAM

class GuidedBackpropCAM(BaseTemporalCAM):
    """
    Guided backpropagation saliency over time for SNNs.

    Returns a (B,T,H,W) saliency map at input resolution by backpropagating
    from the selected class score to the input, while:
      - clamping negative gradients to zero, and
      - gating by positive activations at non-linearities (LIFNode).

    This can be elementwise-multiplied with other CAMs (already upsampled to input size)
    to obtain guided variants (e.g., guided Grad-CAM).
    """
    def __init__(self, model: nn.Module):
        # target_layer is unused; pass model for BaseTemporalCAM signature
        super().__init__(model, target_layer=model)
        self._fwd_outs: Dict[int, torch.Tensor] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    def _register_hooks(self):
        self._fwd_outs.clear()
        self._handles = []

        def make_fwd_hook(key: int):
            def fwd_hook(mod, inp, out):
                # Save forward output for gating; keep detached to avoid ref cycles
                self._fwd_outs[key] = out.detach()
            return fwd_hook

        def make_bwd_hook(key: int):
            def bwd_hook(mod, grad_input, grad_output):
                # grad_input is a tuple; clamp negative entries and gate by positive activation
                if not grad_input or grad_input[0] is None:
                    return grad_input
                gi = grad_input[0]
                gi = gi.clamp_min(0.0)
                act = self._fwd_outs.get(key, None)
                if act is not None:
                    # Gate by positive activations (out > 0)
                    try:
                        mask = (act > 0).to(dtype=gi.dtype)
                        # Shape-align if needed (common case: identical shapes)
                        if mask.shape == gi.shape:
                            gi = gi * mask
                    except Exception:
                        pass
                # Rebuild tuple (keep other entries unchanged)
                gi_list = list(grad_input)
                gi_list[0] = gi
                return tuple(gi_list)
            return bwd_hook

        for m in self.model.modules():
            if isinstance(m, neuron.LIFNode):
                key = id(m)
                self._handles.append(m.register_forward_hook(make_fwd_hook(key)))
                # Use full backward hook to override grad_input
                self._handles.append(m.register_full_backward_hook(make_bwd_hook(key)))

    def _remove_hooks(self):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles = []
        self._fwd_outs.clear()

    def _compute_raw(self, x: torch.Tensor, class_idx: Optional[int]) -> torch.Tensor:
        """
        Compute guided backprop saliency at input resolution.
        Returns (B,T,H,W).
        """
        # Prepare
        self._register_hooks()
        self.model.zero_grad(set_to_none=True)
        functional.reset_net(self.model)

        x = x.detach().clone().requires_grad_(True)          # (T,B,C,H,W)
        with torch.enable_grad():
            outputs = self.model(x)                          # (T,B,K)

        T, B, K = outputs.shape
        if class_idx is None:
            # Use time-summed logits
            summed = outputs.sum(dim=0)                      # (B,K)
            cls = summed.argmax(dim=1)                       # (B,)
        else:
            cls = torch.full((B,), int(class_idx), device=outputs.device, dtype=torch.long)

        # Scalar objective: sum over batch the time-summed class scores
        target = outputs.sum(dim=0).gather(1, cls.view(-1, 1)).sum()
        target.backward()

        # Extract input grads and collapse channel dimension
        grads = x.grad                                        # (T,B,C,H,W)
        self._remove_hooks()
        functional.reset_net(self.model)

        # Positive part, sum across channels -> (T,B,H,W)
        grads_pos = F.relu(grads).sum(dim=2)
        # Reorder to (B,T,H,W)
        cams = grads_pos.permute(1, 0, 2, 3)

        # Already at input resolution; BaseTemporalCAM will optionally normalize
        return cams