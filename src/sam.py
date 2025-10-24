import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union

from models.base_snn import BaseSNN

import snntorch as snn



class SpikeActivationMap:
    """
    Spike Activation Map (SAM) for SNNs.

    - Class-agnostic visualization: no labels, no backprop.
    - Uses only spike outputs from a target spiking layer across time.
    - Efficient recurrence for temporal spike contribution (NCS).

    Usage:
      sam = SpikeActivationMap(model, target_layer)
      cams = sam.generate(x, gamma=0.5, normalize_per_timestep=True, upsample_to_input=True)  # (B, T, H, W)
    """
    def __init__(
        self,
        model: BaseSNN,
        target_layer: nn.Module,
    ):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer

        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._spike_traces: List[torch.Tensor] = []
        self._register_hooks()

    def _clear(self):
        self._spike_traces.clear()

    def remove_hooks(self):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()

    def _register_hooks(self):
        def fwd_hook(module, inputs, output):
            # Some spiking modules may return (spk, mem). Grab the spikes.
            spk = output[0] if isinstance(output, (tuple, list)) else output
            # Expect shape (B, C, H, W)
            self._spike_traces.append(spk.detach())

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

    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,
        gamma: float = 0.5,
        normalize_per_timestep: bool = True,
        upsample_to_input: bool = True,
        return_numpy: bool = False,
    ) -> Union[torch.Tensor, "np.ndarray"]:
        """
        x: (B, T, C, H, W)
        Returns per-timestep SAM maps: (B, T, H, W) at input resolution if upsample_to_input=True.

        Notes:
        - target_layer should be a spiking layer whose forward emits spikes (e.g., snntorch.Leaky).
        - Works with batched inputs; processes all samples jointly.
        """
        assert x.dim() == 5, "Expected input of shape (B, T, C, H, W)"
        device = next(self.model.parameters()).device
        x = x.to(device)

        self._clear()
        # One forward pass collects T spike tensors
        _ = self.model(x)  # (B, T, num_classes) — not used

        assert len(self._spike_traces) > 0, "No spikes captured. Is target_layer a spiking layer?"
        # Stack in time order: (B, T, C, Hf, Wf)
        spikes = torch.stack(self._spike_traces, dim=1)
        B, T, C, Hf, Wf = spikes.shape

        # Recurrence coefficient
        E = torch.exp(torch.tensor(-gamma, device=spikes.device, dtype=spikes.dtype))

        # Compute NCS and SAM per time step
        cams = torch.zeros(B, T, Hf, Wf, device=spikes.device, dtype=spikes.dtype)
        N_prev = torch.zeros(B, C, Hf, Wf, device=spikes.device, dtype=spikes.dtype)

        for t in range(T):
            S_t = spikes[:, t]  # (B, C, Hf, Wf), assumed non-negative (0/1 spikes)
            # M_t = sum_k N_t^k * S_t^k; but N_t uses only previous spikes
            # Using N_prev that represents N_{t} before adding S_{t-1} decay and such, we must produce M_t with NCS built from spikes up to t-1:
            # At t=0, N_prev=0 -> M_0=0
            M_t = (N_prev * S_t).sum(dim=1)  # (B, Hf, Wf)
            cams[:, t] = M_t
            # Update N for next step: N_{t+1} = E * (N_t + S_t)
            N_prev = E * (N_prev + S_t)

        # Optional normalization per timestep
        if normalize_per_timestep:
            cams = self._normalize_per_timestep(cams)

        # Optional upsample to input resolution
        if upsample_to_input:
            H_in, W_in = x.shape[-2], x.shape[-1]
            cams = self._upsample_to_input(cams, (H_in, W_in))

        if return_numpy:
            return cams.detach().cpu().numpy()
        return cams  # (B, T, H, W)
