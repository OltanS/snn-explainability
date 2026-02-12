import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import copy
from spikingjelly.activation_based import functional, neuron
from src.cams.base_temporal_cam import BaseTemporalCAM
from src.noisy_node import noise_injection

class MCAblationCAMTransient(BaseTemporalCAM):
    def __init__(self, model: nn.Module, target_lif_layer: nn.Module, batch_size: int=16, k_trials: int=50, noise_type: str = "surrogate"):
        super().__init__(model, target_lif_layer)
        self.channel_batch_size = batch_size
        self.k_trials = k_trials
        self._importance = None
        self.convergence_data = {}
        self.noise_type = noise_type

        all_neurons = [m for m in self.model.modules() if isinstance(m, neuron.LIFNode) or isinstance(m, neuron.IFNode)]
        t_idx = all_neurons.index(target_lif_layer)
        self._noise_layers = all_neurons[t_idx:]

    def _run_trials(self, x_rep: torch.Tensor, class_idx: int, k: int, scaling: torch.Tensor, target_layer: nn.Module, base_seed: int = 0):
        device = x_rep.device
        B_rep = scaling.shape[0]
        scores = torch.zeros(k, B_rep, device=device)

        def scale_hook(module, inp):
            x_in = inp[0]
            if x_in.dim() == 5:      # (T,B,C,H,W)
                scale = scaling.view(1, B_rep, -1, 1, 1)
            elif x_in.dim() == 4:    # (B,C,H,W)
                scale = scaling.view(B_rep, -1, 1, 1)
            else:
                raise RuntimeError(f"Unexpected target_layer input dim: {x_in.shape}")
            return (x_in * scale,)

        scale_handle = target_layer.register_forward_pre_hook(scale_hook)
        try:
            for i in range(k):
                # Common-random-numbers pairing: caller uses same base_seed for baseline and ablations
                seed = int(base_seed + i)
                torch.manual_seed(seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(seed)

                out = self.model(x_rep)  # (T,B_rep,K) or similar
                scores[i] = out.sum(dim=0)[:, class_idx]
                functional.reset_net(self.model)
        finally:
            scale_handle.remove()

        running = torch.cumsum(scores, dim=0) / torch.arange(1, k + 1, device=device).view(-1, 1)
        return running[-1], running

    def _capture_acts(self, x: torch.Tensor, target_layer: nn.Module):
        acts = None
        def hook(module, inp):
            nonlocal acts
            acts = inp[0].detach().clone()
        h = target_layer.register_forward_pre_hook(hook)
        try:
            out = self.model(x)
        finally:
            h.remove()
        functional.reset_net(self.model)
        return acts, out

    def compute_importance(self, x: torch.Tensor, k_trials: int = 50, class_idx: int = None, base_seed: int = 1234):
        with noise_injection(self.model, noise_type=self.noise_type, target_layer=self.target_layer) as noisy_target:
            with torch.no_grad():
                acts, outputs = self._capture_acts(x, target_layer=noisy_target)
                T, B, C, H, W = acts.shape
                assert B == 1
                device = acts.device
                del acts

                sums = outputs.sum(dim=0)
                if class_idx is None:
                    class_idx = int(sums.argmax(dim=1).item())

                baseline_scaling = torch.ones(1, C, device=device)
                baseline_final, baseline_running = self._run_trials(x, class_idx, k_trials, baseline_scaling, target_layer=noisy_target, base_seed=base_seed)
                self.convergence_data["baseline"] = baseline_running[:, 0].cpu().numpy()

                importance = torch.zeros(C, device=device)
                for start in range(0, C, self.channel_batch_size):
                    end = min(start + self.channel_batch_size, C)
                    G = end - start
                    scaling = torch.ones(G, C, device=device)
                    for i, ch in enumerate(range(start, end)):
                        scaling[i, ch] = 0.0

                    x_rep = x.repeat(1, G, 1, 1, 1)
                    ablated_final, ablated_running = self._run_trials(x_rep, class_idx, k_trials, scaling, target_layer=noisy_target, base_seed=base_seed)

                    drops = baseline_final - ablated_final
                    importance[start:end] = drops
                    for i, ch in enumerate(range(start, end)):
                        self.convergence_data[f"ch_{ch}"] = ablated_running[:, i].cpu().numpy()

                self._importance = importance
                return importance
    
    def _compute_raw(self, x: torch.Tensor, class_idx: int) -> torch.Tensor:
        # Ensure importance exists (computed under the noisy model)
        if self._importance is None:
            self.compute_importance(x, k_trials=self.k_trials, class_idx=class_idx)

        # Capture activations under the same noisy model to match the importance definition
        with noise_injection(self.model, noise_type=self.noise_type, target_layer=self.target_layer) as noisy_target:
            with torch.no_grad():
                acts, _ = self._capture_acts(x, target_layer=noisy_target)  # (T,B,C,Hf,Wf)

        imp = self._importance.view(1, 1, -1, 1, 1)   # (1,1,C,1,1)
        cams = (acts * imp).sum(dim=2)                # (T,B,Hf,Wf)
        cams = cams.permute(1, 0, 2, 3)               # (B,T,Hf,Wf)
        return F.relu(cams)

    def compute_convergence(self, x: torch.Tensor, class_idx: int = None, check_every: int = 10) -> dict:
        """
        Computes CAMs at intermediate steps to analyze convergence.
        Returns a dictionary mapping k_trials -> cam_tensor.
        """
        # 1) Capture activations (use the same noisy model as convergence estimates)
        with noise_injection(self.model, noise_type=self.noise_type, target_layer=self.target_layer) as noisy_target:
            with torch.no_grad():
                acts, outputs = self._capture_acts(x, target_layer=noisy_target)  # (T,B,C,H,W)
                T, B, C, H, W = acts.shape
                assert B == 1, "Batch size must be 1 for convergence check"
                device = acts.device

                sums = outputs.sum(dim=0)
                if class_idx is None:
                    class_idx = int(sums.argmax(dim=1).item())

                # 2) Run Trials (Baseline & Ablation)
                baseline_scaling = torch.ones(1, C, device=device)
                _, baseline_running = self._run_trials(
                    x, class_idx, self.k_trials, baseline_scaling, target_layer=noisy_target
                )  # (K,1)

                ablated_running_all = torch.zeros(self.k_trials, C, device=device)
                for start in tqdm(range(0, C, self.channel_batch_size), desc="Ablating channels (Convergence)"):
                    end = min(start + self.channel_batch_size, C)
                    G = end - start

                    scaling = torch.ones(G, C, device=device)
                    for i, ch in enumerate(range(start, end)):
                        scaling[i, ch] = 0.0

                    x_rep = x.repeat(1, G, 1, 1, 1)
                    _, ablated_running = self._run_trials(
                        x_rep, class_idx, self.k_trials, scaling, target_layer=noisy_target
                    )  # (K,G)
                    ablated_running_all[:, start:end] = ablated_running

        # 3) Compute CAMs at checkpoints
        results = {}
        importance_all = baseline_running - ablated_running_all  # (K,C)

        ks = list(range(check_every, self.k_trials + 1, check_every))
        if self.k_trials % check_every != 0 and self.k_trials not in ks:
            ks.append(self.k_trials)

        for k in ks:
            imp = importance_all[k - 1]  # (C,)
            cams = (acts * imp.view(1, 1, -1, 1, 1)).sum(dim=2)  # (T,1,H,W)
            cams = cams.permute(1, 0, 2, 3)  # (1,T,H,W)
            results[k] = F.relu(cams).cpu()

        return results