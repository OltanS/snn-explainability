import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List

from spikingjelly.activation_based import functional, layer
from src.cams.base_temporal_cam import BaseTemporalCAM
from src.noisy_node import noise_injection

class SNNNoiseGradCAM(BaseTemporalCAM):
    def __init__(self, model: nn.Module, target_layer: nn.Module, k_trials: int = 100, batch_size: int = 16, noise_type: str = "surrogate"):
        super().__init__(model, target_layer)
        self.k_trials = k_trials
        self.batch_size = batch_size
        self.noise_type = noise_type

    def _compute_raw(self, x: torch.Tensor, class_idx: Optional[int]) -> torch.Tensor:
        acts = None
        grads = None
        T, B, C_in, H, W = x.shape

        # Determine target classes
        if class_idx is None:
            # Run a clean pass to determine predicted class
            with torch.no_grad():
                functional.reset_net(self.model)
                out_clean = self.model(x)
                target_classes = out_clean.mean(dim=0).argmax(dim=1)  # (B,)
                functional.reset_net(self.model)

        else:
            target_classes = torch.full((B,), int(class_idx), device=x.device, dtype=torch.long)
        
        sum_cams = None

        with noise_injection(self.model, noise_type=self.noise_type, target_layer=self.target_layer):
            trials_run = 0
            while trials_run < self.k_trials:
                current_batch_k = min(self.batch_size, self.k_trials - trials_run)
                
                # Repeat x: (T, B*k, C, H, W) -> [Batch1, Batch2, ..., Batch1, Batch2, ...]
                x_rep = x.repeat(1, current_batch_k, 1, 1, 1)
                x_rep = x_rep.detach().requires_grad_(True)
                
                acts = None
                grads = None
                
                def fwd_hook(module, inp, out):
                    nonlocal acts
                    acts = out
                    out.retain_grad()
                    def grad_hook(g):
                        nonlocal grads
                        grads = g
                    out.register_hook(grad_hook)

                handle = self.target_layer.register_forward_hook(fwd_hook)
                
                try:
                    self.model.zero_grad(set_to_none=True)
                    functional.reset_net(self.model)
                    
                    with torch.enable_grad():
                        outputs = self.model(x_rep)  # (T, B*k, K)
                        
                        # Prepare targets
                        # target_classes: (B,) -> repeat to (B*k,)
                        target_classes_rep = target_classes.repeat(current_batch_k)
                        
                        T_out, Bk, K = outputs.shape
                        
                        # Gather scores
                        target_score = outputs.gather(2, target_classes_rep.view(1, Bk, 1).expand(T_out, Bk, 1)).mean()
                        target_score.backward()
                    
                    assert acts is not None, "Missing activations."
                    assert grads is not None, "Missing gradients."
                    
                    # Compute weights per trial
                    w = grads.mean(dim=(-2, -1))  # (T, Bk, C)
                    
                    # Reshape to (T, k, B, C)
                    w = w.view(T_out, current_batch_k, B, -1)
                    a = acts.view(T_out, current_batch_k, B, acts.shape[2], acts.shape[3], acts.shape[4])
                    
                    # Compute CAMs for this batch of trials immediately
                    # w: (T, k, B, C) -> (T, k, B, C, 1, 1)
                    # a: (T, k, B, C, H, W)
                    batch_cams = (w.unsqueeze(-1).unsqueeze(-1) * a).sum(dim=3)
                    batch_cams = F.relu(batch_cams) # Apply ReLU per trial
                    
                    # Sum over k to accumulate
                    batch_cams_sum = batch_cams.sum(dim=1) # (T, B, H, W)
                    
                    if sum_cams is None:
                        sum_cams = batch_cams_sum.detach()
                    else:
                        sum_cams += batch_cams_sum.detach()
                        
                finally:
                    handle.remove()
                    functional.reset_net(self.model)
                    # Clear graph
                    acts = None
                    grads = None
                
                trials_run += current_batch_k

        # Average
        avg_cams = sum_cams / self.k_trials
        avg_cams = avg_cams.permute(1, 0, 2, 3) # (B, T, Hf, Wf)
        return avg_cams
    
    def compute_convergence(self, x: torch.Tensor, class_idx: Optional[int] = None, check_every: int = 10) -> dict:
        """
        Computes CAMs at intermediate steps to analyze convergence.
        Returns a dictionary mapping k_trials -> cam_tensor.
        """
        results = {}
        T, B, C_in, H, W = x.shape

        # Determine target classes
        if class_idx is None:
            with torch.no_grad():
                functional.reset_net(self.model)
                out_clean = self.model(x)
                target_classes = out_clean.mean(dim=0).argmax(dim=1)
                functional.reset_net(self.model)
        else:
            target_classes = torch.full((B,), int(class_idx), device=x.device, dtype=torch.long)
        
        sum_cams = None
        
        next_check = check_every

        with noise_injection(self.model, noise_type=self.noise_type, target_layer=self.target_layer):
            trials_run = 0
            while trials_run < self.k_trials:
                current_batch_k = min(self.batch_size, self.k_trials - trials_run)
                
                # Repeat x
                x_rep = x.repeat(1, current_batch_k, 1, 1, 1)
                x_rep = x_rep.detach().requires_grad_(True)
                
                acts = None
                grads = None
                
                def fwd_hook(module, inp, out):
                    nonlocal acts
                    acts = out
                    out.retain_grad()
                    def grad_hook(g):
                        nonlocal grads
                        grads = g
                    out.register_hook(grad_hook)

                handle = self.target_layer.register_forward_hook(fwd_hook)
                
                try:
                    self.model.zero_grad(set_to_none=True)
                    functional.reset_net(self.model)
                    
                    with torch.enable_grad():
                        outputs = self.model(x_rep)
                        target_classes_rep = target_classes.repeat(current_batch_k)
                        T_out, Bk, K = outputs.shape
                        target_score = outputs.gather(2, target_classes_rep.view(1, Bk, 1).expand(T_out, Bk, 1)).mean()
                        target_score.backward()
                    
                    # Compute weights per trial
                    w = grads.mean(dim=(-2, -1))  # (T, Bk, C)
                    w = w.view(T_out, current_batch_k, B, -1)
                    a = acts.view(T_out, current_batch_k, B, acts.shape[2], acts.shape[3], acts.shape[4])
                    
                    # Compute CAMs for this batch
                    batch_cams = (w.unsqueeze(-1).unsqueeze(-1) * a).sum(dim=3)
                    batch_cams = F.relu(batch_cams)
                    
                    # Sum over k
                    batch_cams_sum = batch_cams.sum(dim=1)
                    
                    if sum_cams is None:
                        sum_cams = batch_cams_sum.detach()
                    else:
                        sum_cams += batch_cams_sum.detach()
                        
                finally:
                    handle.remove()
                    functional.reset_net(self.model)
                    acts = None
                    grads = None
                
                trials_run += current_batch_k

                # Checkpoint
                if trials_run >= next_check or trials_run == self.k_trials:
                    avg_cams = sum_cams / trials_run
                    # Permute to (B, T, H, W)
                    res = avg_cams.permute(1, 0, 2, 3).cpu()
                    results[trials_run] = res

                    while next_check <= trials_run:
                        next_check += check_every
    
        return results