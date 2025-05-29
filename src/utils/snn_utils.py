import torch

def compute_raw_mem(out_spikes: torch.Tensor,
                    out_mem:    torch.Tensor,
                    threshold:  float = 1.0
                   ) -> torch.Tensor:
    """
    Reconstruct the pre-reset membrane potential at each timestep:
      raw_mem[t] = out_mem[t] + out_spikes[t] * threshold

    Args:
      out_spikes: (B, T, C) spikes from final layer
      out_mem:    (B, T, C) post-reset membrane from final layer
      threshold:  firing threshold (default=1.0)
    Returns:
      raw_mem:    (B, T, C) “never-fired” membrane
    """
    return out_mem + out_spikes * threshold

def compute_delta_raw_mem(raw_mem: torch.Tensor,
                          keep_first_zero: bool = True
                         ) -> torch.Tensor:
    """
    Compute Δ raw_mem[t] = raw_mem[t] − raw_mem[t-1].
    If keep_first_zero, pads a zero at t=0 so out is (B,T,C), else returns (B,T-1,C).

    In a non-spiking network, this is equivalent to computing the difference
    between consecutive timesteps of the class outputs.
    """
    delta = raw_mem[:, 1:] - raw_mem[:, :-1]   # (B, T-1, C)
    if keep_first_zero:
        zero = torch.zeros_like(raw_mem[:, :1])  # (B,1,C)
        delta = torch.cat([zero, delta], dim=1)   # (B, T, C)
    return delta