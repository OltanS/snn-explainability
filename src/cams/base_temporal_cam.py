import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Union

from src.cams.cam_utils import normalize_per_timestep, upsample_cams
from spikingjelly.activation_based import functional

class BaseTemporalCAM(ABC):
    """
    Unified interface for temporal CAM methods.
    generate(...) returns (B,T,H,W) unless squeeze_batch=True and B==1 -> (T,H,W).
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer


    @abstractmethod
    def _compute_raw(self, x: torch.Tensor, class_idx: Optional[int]) -> torch.Tensor:
        """
        Return raw cams at feature resolution: (B,T,Hf,Wf).
        May perform forward/backward/hooking internally.
        """

    def generate(
        self,
        x: torch.Tensor,                 # (T,B,C,H,W)
        class_idx: Optional[int] = None,
        normalize: bool = True,
        upsample: bool = True,
        squeeze_batch: bool = True,
        return_numpy: bool = False,
    ) -> Union[torch.Tensor, "np.ndarray"]:
        assert x.dim() == 5
        device = next(self.model.parameters()).device
        x = x.to(device)
        cams = self._compute_raw(x, class_idx)        # (B,T,Hf,Wf)
        if normalize:
            cams = normalize_per_timestep(cams)
        if upsample:
            cams = upsample_cams(cams, (x.shape[-2], x.shape[-1]))
        
        functional.reset_net(self.model)
        
        if squeeze_batch and cams.shape[0] == 1:
            cams = cams[0]
        if return_numpy:
            return cams.detach().cpu().numpy()
        return cams