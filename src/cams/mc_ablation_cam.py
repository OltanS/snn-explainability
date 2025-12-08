import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from spikingjelly.activation_based import functional, neuron
from src.cams.base_temporal_cam import BaseTemporalCAM

class MCAblationCAM(BaseTemporalCAM):
    def __init__(self, model: nn.Module, target_lif_layer: nn.Module, channel_batch_size: int = 16, k_trials: int = 50, use_post_hook: bool = False):
        super().__init__(model, target_lif_layer)
        self.channel_batch_size = channel_batch_size
        self.k_trials = k_trials
        self._importance = None
        self._activations = None
        self.convergence_data = {}
        self.use_post_hook = use_post_hook

        all_lifs = [m for m in self.model.modules() if isinstance(m, neuron.LIFNode) or isinstance(m, neuron.IFNode)]
        t_idx = all_lifs.index(target_lif_layer)
        self._noise_layers = all_lifs[t_idx:]

    def _run_trials(self, x_rep: torch.Tensor, class_idx: int, k: int, scaling: torch.Tensor):
        device = x_rep.device
        B_rep = scaling.shape[0]
        scores = torch.zeros(k, B_rep, device=device)
        def noise_hook(module, inp):
            v_thr = getattr(module, 'v_threshold', 1.0)
            return (inp[0] + torch.rand_like(inp[0]) * v_thr, )
        noise_handles = [l.register_forward_pre_hook(noise_hook) for l in self._noise_layers]
        if self.use_post_hook:
            def scale_hook(module, inp, out):
                return out * scaling.view(1, B_rep, -1, 1, 1)
            scale_handle = self.target_layer.register_forward_hook(scale_hook)
        else:
            def scale_hook(module, inp):
                return (inp[0] * scaling.view(1, B_rep, -1, 1, 1), )
            scale_handle = self.target_layer.register_forward_pre_hook(scale_hook)
        try:
            for i in range(k):
                out = self.model(x_rep)  # (T,B_rep,K)
                scores[i] = out.sum(dim=0)[:, class_idx]
                functional.reset_net(self.model)
        finally:
            for h in noise_handles: h.remove()
            scale_handle.remove()

        running = torch.cumsum(scores, dim=0) / torch.arange(1, k + 1, device=device).view(-1, 1)
        return running[-1], running

    def _capture_acts(self, x: torch.Tensor):

        acts = None

        if self.use_post_hook:
            def hook(module, inp, out):
                nonlocal acts
                acts = out.detach().clone()
            h = self.target_layer.register_forward_hook(hook)
        else:
            def hook(module, inp):
                nonlocal acts
                acts = inp[0].detach().clone()
            h = self.target_layer.register_forward_pre_hook(hook)
        
        try:
            out = self.model(x)
        finally:
            h.remove()
            
        functional.reset_net(self.model)
        return acts, out

    def compute_importance(self, x: torch.Tensor, k_trials: int = 50, class_idx: int = None):
        with torch.no_grad():
            acts, outputs = self._capture_acts(x)
            self._activations = acts          # (T,B,C,H,W)
            T, B, C, H, W = acts.shape
            assert B == 1
            sums = outputs.sum(dim=0)
            if class_idx is None:
                class_idx = int(sums.argmax(dim=1).item())
            baseline_scaling = torch.ones(1, C, device=acts.device)
            baseline_final, baseline_running = self._run_trials(x, class_idx, k_trials, baseline_scaling)
            self.convergence_data['baseline'] = baseline_running[:, 0].cpu().numpy()
            importance = torch.zeros(C, device=acts.device)
            for start in tqdm(range(0, C, self.channel_batch_size), desc="Ablating channels", disable=True):
                end = min(start + self.channel_batch_size, C)
                G = end - start
                scaling = torch.ones(G, C, device=acts.device)
                for i, ch in enumerate(range(start, end)):
                    scaling[i, ch] = 0.0
                x_rep = x.repeat(1, G, 1, 1, 1)
                ablated_final, ablated_running = self._run_trials(x_rep, class_idx, k_trials, scaling)
                drops = baseline_final - ablated_final  # (G,)
                importance[start:end] = drops
                for i, ch in enumerate(range(start, end)):
                    self.convergence_data[f'ch_{ch}'] = ablated_running[:, i].cpu().numpy()
            self._importance = importance
            return importance

    def _compute_raw(self, x: torch.Tensor, class_idx: int) -> torch.Tensor:
        if self._importance is None:
            self.compute_importance(x, k_trials=self.k_trials, class_idx=class_idx)
        acts = self._activations          # (T,B,C,Hf,Wf)
        imp = self._importance.view(1, 1, -1, 1, 1)
        weighted = acts * imp
        cams = weighted.sum(dim=2)        # (T,B,Hf,Wf)
        cams = cams.permute(1, 0, 2, 3)   # (B,T,Hf,Wf)
        cams = F.relu(cams)
        return cams