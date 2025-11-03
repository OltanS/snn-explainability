import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union

from spikingjelly.activation_based import neuron, surrogate, functional

class SpikeActivationMap:
    """
    Spike Activation Map (SAM) for SNNs.

    - Class-agnostic visualization: no labels, no backprop.
    - Uses only spike outputs from a target spiking layer across time.
    - Efficient recurrence for temporal spike contribution (NCS).

    Expected input (SpikingJelly, step_mode='m'): x of shape (T, B, C, H, W)
    Returns: (T, B, H, W)
    """
    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
    ):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer

        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._spike_traces: None
        self._register_hooks()

    def _clear(self):
        self._spike_traces = None

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
            spk = output
            # Expect shape (B, C, H, W)
            self._spike_traces = spk.detach()

        self._handles.append(self.target_layer.register_forward_hook(fwd_hook))

    @staticmethod
    def _normalize_per_timestep(cams: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        # cams: (B, T, H, W)
        T, B, H, W = cams.shape
        flat = cams.view(T, B, -1)
        mins = flat.min(dim=-1).values.view(T, B, 1, 1)
        maxs = flat.max(dim=-1).values.view(T, B, 1, 1)
        denom = (maxs - mins).clamp_min(eps)
        return ((cams - mins) / denom).clamp(0, 1)

    @staticmethod
    def _upsample_to_input(cams_bt_hw: torch.Tensor, input_hw: Tuple[int, int]) -> torch.Tensor:
        T, B, Hf, Wf = cams_bt_hw.shape
        cams_b1hw = cams_bt_hw.view(B * T, 1, Hf, Wf)
        cams_b1hw = F.interpolate(cams_b1hw, size=input_hw, mode="bilinear", align_corners=False)
        return cams_b1hw.view(T, B, *input_hw)

    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor, # (T, B, C, H, W)
        gamma: float = 0.5,
        normalize_per_timestep: bool = True,
        upsample_to_input: bool = True,
        return_numpy: bool = False,
    ) -> Union[torch.Tensor, "np.ndarray"]:
        """
        Returns per-timestep SAM maps: (B, T, H, W) at input resolution if upsample_to_input=True.
        """
        assert x.dim() == 5, "Expected input of shape (T, B, C, H, W)"
        device = next(self.model.parameters()).device
        x = x.to(device)

        self._clear()
        # One forward pass collects T spike tensors
        _ = self.model(x)  # (B, T, num_classes) — not used

        assert self._spike_traces is not None, "No spikes captured. Is target_layer a spiking layer?"
        spk_seq = self._spike_traces                # (T, B, C, Hf, Wf)

        T, B, C, Hf, Wf = spk_seq.shape

        # Recurrence coefficient
        E = torch.exp(torch.tensor(-gamma, device=spk_seq.device, dtype=spk_seq.dtype))

        # Compute NCS and SAM per time step
        cams = torch.zeros(T, B, Hf, Wf, device=spk_seq.device, dtype=spk_seq.dtype)
        N_prev = torch.zeros(B, C, Hf, Wf, device=spk_seq.device, dtype=spk_seq.dtype)

        for t in range(T):
            S_t = spk_seq[t]  # (B, C, Hf, Wf), assumed non-negative (0/1 spikes)
            # M_t = sum_k N_t^k * S_t^k; but N_t uses only previous spikes
            # Using N_prev that represents N_{t} before adding S_{t-1} decay and such, we must produce M_t with NCS built from spikes up to t-1:
            # At t=0, N_prev=0 -> M_0=0
            M_t = (N_prev * S_t).sum(dim=1)  # (B, Hf, Wf)
            cams[t] = M_t
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
        functional.reset_net(self.model)
        cams = cams.transpose(1, 0)  # (B, T, H, W)
        return cams  # (B, T, H, W)
