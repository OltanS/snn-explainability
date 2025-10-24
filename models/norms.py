import torch.nn as nn
import snntorch as snn

class _ChannelwiseLN2d(nn.Module):
    def __init__(self, channels: int, eps: float, affine: bool):
        super().__init__()
        self.ln = nn.LayerNorm(channels, eps=eps, elementwise_affine=affine)
    def forward(self, x):
        # x: (N, C, H, W)
        x = x.permute(0, 2, 3, 1)        # (N, H, W, C)
        x = self.ln(x)
        return x.permute(0, 3, 1, 2).contiguous()     # (N, C, H, W)

class _TimeSharedWrapper(nn.Module):
    """
    Wrap a time-shared normalization module and expose __getitem__(t)
    to mimic the BatchNormTT API.
    """
    def __init__(self, module: nn.Module):
        super().__init__()
        self._module = module

    def __getitem__(self, idx):
        return self._module


def _norm2d_bntt(channels: int, time_steps: int, eps: float, affine: bool) -> nn.Module:
    return snn.BatchNormTT2d(channels, time_steps=time_steps, eps=eps, affine=affine)

def _norm2d_batchnorm(channels: int, eps: float, affine: bool) -> nn.Module:
    return _TimeSharedWrapper(nn.BatchNorm2d(channels, eps=eps, affine=affine))

def _norm2d_layernorm(channels: int, eps: float, affine: bool) -> nn.Module:
    return _TimeSharedWrapper(_ChannelwiseLN2d(channels, eps, affine))

def _norm2d_none() -> nn.Module:
    return _TimeSharedWrapper(nn.Identity())

def _norm1d_bntt(features: int, time_steps: int, eps: float, affine: bool) -> nn.Module:
    return snn.BatchNormTT1d(features, time_steps=time_steps, eps=eps, affine=affine)

def _norm1d_batchnorm(features: int, eps: float, affine: bool) -> nn.Module:
    return _TimeSharedWrapper(nn.BatchNorm1d(features, eps=eps, affine=affine))

def _norm1d_layernorm(features: int, eps: float, affine: bool) -> nn.Module:
    # True LayerNorm over the last dimension (B, F)
    return _TimeSharedWrapper(nn.LayerNorm(normalized_shape=features, eps=eps, elementwise_affine=affine))

def _norm1d_none() -> nn.Module:
    return _TimeSharedWrapper(nn.Identity())

def make_norm2d_factory(norm_type: str, time_steps: int):
    """
    Returns a callable: factory(channels: int, eps=1e-5, affine=True) -> nn.Module
    Supported norm_type: "bntt", "batchnorm", "layernorm", "none"
    """
    key = norm_type.lower()
    def factory(channels: int, eps: float = 1e-5, affine: bool = True) -> nn.Module:
        if key == "bntt":
            return _norm2d_bntt(channels, time_steps=time_steps, eps=eps, affine=affine)
        elif key == "batchnorm":
            return _norm2d_batchnorm(channels, eps=eps, affine=affine)
        elif key == "layernorm":
            return _norm2d_layernorm(channels, eps=eps, affine=affine)
        elif key == "none":
            return _norm2d_none()
        else:
            raise ValueError(f"Unknown 2D norm_type '{norm_type}'. Expected one of: bntt, batchnorm, layernorm, none.")
    return factory

def make_norm1d_factory(norm_type: str, time_steps: int):
    """
    Returns a callable: factory(features: int, eps=1e-5, affine=True) -> nn.Module
    Supported norm_type: "bntt", "batchnorm", "layernorm", "none"
    """
    key = norm_type.lower()
    def factory(features: int, eps: float = 1e-5, affine: bool = True) -> nn.Module:
        if key == "bntt":
            return _norm1d_bntt(features, time_steps=time_steps, eps=eps, affine=affine)
        elif key == "batchnorm":
            return _norm1d_batchnorm(features, eps=eps, affine=affine)
        elif key == "layernorm":
            return _norm1d_layernorm(features, eps=eps, affine=affine)
        elif key == "none":
            return _norm1d_none()
        else:
            raise ValueError(f"Unknown 1D norm_type '{norm_type}'. Expected one of: bntt, batchnorm, layernorm, none.")
    return factory
