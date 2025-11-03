import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union

from spikingjelly.activation_based import neuron, surrogate, functional, layer


class SNNGradCAM:
    """
    Grad-CAM for Spiking Neural Networks (per time-step, SpikingJelly step_mode='m').
    
    Expected input: (T, B, C, H, W)
    Returns: (B, T, H, W)
    """
    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        relu: bool = True,
    ):
        self.model = model
        self.target_layer = target_layer
        self.relu = relu

        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._acts = None
        self._grads = None
        self._register_hooks()

    def _clear(self):
        self._acts = None
        self._grads = None

    def remove_hooks(self):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()

    def _register_hooks(self):
        def fwd_hook(module, inputs, output):
            # output: (B, C, H, W) for a single time step call
            self._acts = output
            output.retain_grad()
            # Attach a grad hook to this specific output tensor to capture its gradient
            def grad_hook(grad):
                self._grads = grad
            output.register_hook(grad_hook)

        self._handles.append(self.target_layer.register_forward_hook(fwd_hook))

    @staticmethod
    def _normalize_per_timestep(cams: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        # cams: (B, T, H, W)
        B, T, H, W = cams.shape
        flat = cams.view(B, T, -1)
        mins = flat.min(dim=-1).values.view(B, T, 1, 1)
        maxs = flat.max(dim=-1).values.view(B, T, 1, 1)
        denom = (maxs - mins).clamp_min(eps)
        return ((cams - mins) / denom).clamp(0, 1)

    @staticmethod
    def _upsample_to_input(cams_bt_hw: torch.Tensor, input_hw: Tuple[int, int]) -> torch.Tensor:
        # cams_bt_hw: (B, T, Hf, Wf)
        B, T, Hf, Wf = cams_bt_hw.shape
        cams_b1hw = cams_bt_hw.view(B * T, 1, Hf, Wf)
        cams_b1hw = F.interpolate(cams_b1hw, size=input_hw, mode="bilinear", align_corners=False)
        return cams_b1hw.view(B, T, *input_hw)

    def generate(
        self,
        x: torch.Tensor,  # (T, B, C, H, W)
        class_idx: Optional[Union[int, List[int]]] = None,
        normalize_per_timestep: bool = False,
        upsample_to_input: bool = True,
        return_numpy: bool = False,
    ) -> Union[torch.Tensor, 'np.ndarray']:
        """
        Returns per-timestep CAMs: (B, T, H, W).
        """
        assert x.dim() == 5, "Expected input of shape (T, B, C, H, W)"
        device = next(self.model.parameters()).device
        x = x.to(device)
        x = x.detach().requires_grad_(True)

        self._clear()
        self.model.zero_grad(set_to_none=True)
        functional.reset_net(self.model)

        # Forward: collect activations (and bind grad hooks) per time step
        with torch.enable_grad():
            outputs = self.model(x) # (T, B, K)
        T, B, num_classes = outputs.shape
        assert self._acts is not None, "No activations captured at target layer."

        # Choose class index per-sample if not provided
        # Aggregate scores across time before backprop to produce time-distributed gradients
        if class_idx is None:
            time_agg = outputs.mean(dim=0)
            cls = time_agg.argmax(dim=1)  # (B,)
        else:
            if isinstance(class_idx, int):
                cls = torch.full((B,), int(class_idx), device=outputs.device, dtype=torch.long)
            else:
                cls = torch.tensor(class_idx, device=outputs.device, dtype=torch.long)
                assert cls.shape[0] == B

        # Scalar target for backward
        target_score = outputs.gather(2, cls.view(1, B, 1).expand(T, B, 1)).squeeze(2).mean()
        target_score.backward()

        # Use the last captured act/grad (multi-step): (T, B, C, Hf, Wf)
        acts = self._acts
        grads = self._grads
        assert self._grads is not None, "No gradients captured at target layer."
        assert acts.shape == grads.shape and acts.dim() == 5, "Activation/gradient shape mismatch."

        # Weights per (T, B, C): GAP over spatial dims
        weights = grads.mean(dim=(-2, -1))                              # (T, B, C)
        cams_tbhw = (weights.unsqueeze(-1).unsqueeze(-1) * acts).sum(dim=2)  # (T, B, Hf, Wf)
        cams_bthw = cams_tbhw.permute(1, 0, 2, 3)                        # (B, T, Hf, Wf)

        if self.relu:
            cams_bthw = F.relu(cams_bthw)

        # Normalize per time step (optional)
        if normalize_per_timestep:
            cams_bthw = self._normalize_per_timestep(cams_bthw)

        # Upsample to input resolution if requested
        if upsample_to_input:
            H_in, W_in = x.shape[-2], x.shape[-1]
            cams_bthw = self._upsample_to_input(cams_bthw, (H_in, W_in))

        if return_numpy:
            return cams_bthw.detach().cpu().numpy()
        return cams_bthw  # (B, T, H, W)

def get_default_conv_layer(model: nn.Module) -> nn.Module:
    """Pick the deepest Conv2d by default."""
    convs = [m for m in model.modules() if isinstance(m, layer.Conv2d)]
    if not convs:
        raise ValueError("No SpikingJelly Conv2d layers found.")
    return convs[-1]