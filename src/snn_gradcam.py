import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union

from models.base_snn import BaseSNN

class SNNGradCAM:
    """
    Grad-CAM for Spiking Neural Networks (per time-step).
    
    Usage:
      cam = SNNGradCAM(model, target_layer)
      cams = cam.generate(x, class_idx=None, score_aggregation="mean")  # (B, T, H, W)
    """
    def __init__(
        self,
        model: BaseSNN,
        target_layer: nn.Module,
        relu: bool = True,
    ):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.relu = relu

        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._acts: List[torch.Tensor] = []
        self._grads: List[torch.Tensor] = []
        self._register_hooks()

    def _clear(self):
        self._acts.clear()
        self._grads.clear()

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
            self._acts.append(output)
            # Attach a grad hook to this specific output tensor to capture its gradient
            def grad_hook(grad):
                self._grads.append(grad)
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
        x: torch.Tensor,
        class_idx: Optional[Union[int, List[int]]] = None,
        score_aggregation: str = "mean",   # "mean" or "sum" across time
        normalize_per_timestep: bool = False,
        upsample_to_input: bool = True,
        return_numpy: bool = False,
    ) -> Union[torch.Tensor, 'np.ndarray']:
        """
        x: (B, T, C, H, W)
        Returns per-timestep CAMs: (B, T, H, W) at input resolution if upsample_to_input=True.
        """
        assert x.dim() == 5, "Expected input of shape (B, T, C, H, W)"
        device = next(self.model.parameters()).device
        x = x.to(device)

        self._clear()
        self.model.zero_grad(set_to_none=True)

        # Forward: collect activations (and bind grad hooks) per time step
        outputs = self.model(x)  # (B, T, num_classes)
        B, T, num_classes = outputs.shape
        assert len(self._acts) == T, "Hooked activations count must equal time steps."
        assert len(self._grads) == 0, "Unexpected gradients captured before backward."

        # Choose class index per-sample if not provided
        # Aggregate scores across time before backprop to produce time-distributed gradients
        if class_idx is None:
            time_agg = outputs.mean(dim=1) if score_aggregation == "mean" else outputs.sum(dim=1)
            cls = time_agg.argmax(dim=1)  # (B,)
        else:
            if isinstance(class_idx, int):
                cls = torch.full((B,), int(class_idx), device=outputs.device, dtype=torch.long)
            else:
                cls = torch.tensor(class_idx, device=outputs.device, dtype=torch.long)
                assert cls.shape[0] == B

        # Scalar target for backward
        if score_aggregation == "mean":
            target_score = outputs[torch.arange(B, device=device)[:, None], torch.arange(T, device=device)[None, :], cls[:, None]].mean()
        elif score_aggregation == "sum":
            target_score = outputs[torch.arange(B, device=device)[:, None], torch.arange(T, device=device)[None, :], cls[:, None]].sum()
        else:
            raise ValueError("score_aggregation must be 'mean' or 'sum'.")

        target_score.backward()

        # After backward, grads collected per forward call; convert lists to tensors in time order
        # Each entry: (B, C, Hf, Wf)
        acts = torch.stack(self._acts, dim=1)   # (B, T, C, Hf, Wf)
        grads = torch.stack(self._grads, dim=1) # (B, T, C, Hf, Wf)
        assert acts.shape == grads.shape, "Activation/gradient shape mismatch."

        # Weights per (B, T, C): global-average pooling over spatial dims
        weights = grads.mean(dim=(-2, -1))  # (B, T, C)

        # Weighted sum over channels -> (B, T, Hf, Wf)
        cams = (weights.unsqueeze(-1).unsqueeze(-1) * acts).sum(dim=2)

        if self.relu:
            cams = F.relu(cams)

        # Normalize per time step (optional)
        if normalize_per_timestep:
            cams = self._normalize_per_timestep(cams)

        # Upsample to input resolution if requested
        if upsample_to_input:
            H_in, W_in = x.shape[-2], x.shape[-1]
            cams = self._upsample_to_input(cams, (H_in, W_in))

        if return_numpy:
            return cams.detach().cpu().numpy()
        return cams  # (B, T, H, W)

def get_default_conv_layer(model: nn.Module) -> nn.Module:
    """Pick the deepest Conv2d by default."""
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        raise ValueError("No Conv2d layers found.")
    return convs[-1]