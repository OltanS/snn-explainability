import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.activation_based import functional
from src.cams.base_temporal_cam import BaseTemporalCAM

class AblationCAM(BaseTemporalCAM):
    def __init__(self, model: nn.Module, target_layer: nn.Module, channel_batch_size: int = 16, use_pre_hook: bool = False):
        super().__init__(model, target_layer)
        self.channel_batch_size = channel_batch_size
        self.use_pre_hook = use_pre_hook
        self._activations = None
        self._importance = None

    def _capture_acts(self, x: torch.Tensor):
        acts = None
        
        if self.use_pre_hook:
            # Capture input to the layer (e.g. LIF)
            def hook(module, args):
                nonlocal acts
                # args is a tuple, args[0] is the input tensor
                acts = args[0].detach().clone()
                return None
            h = self.target_layer.register_forward_pre_hook(hook)
        else:
            # Capture output of the layer (e.g. Conv)
            def hook(module, inp, out):
                nonlocal acts
                acts = out.detach().clone()

            h = self.target_layer.register_forward_hook(hook)

        try:
            out = self.model(x)  # (T,B,K)
        finally:
            h.remove()
            
        functional.reset_net(self.model)
        return acts, out

    def compute_importance(self, x: torch.Tensor, class_idx: int = None):
        with torch.no_grad():
            acts, outputs = self._capture_acts(x)
            
            # Normalize to (T,B,C,H,W)
            if acts.dim() == 4:
                acts = acts.unsqueeze(0)
            
            self._activations = acts
            T, B, C, H, W = acts.shape
            assert B == 1, "Batch size 1 expected."

            sums = outputs.sum(dim=0)              # (B,K)
            if class_idx is None:
                class_idx = int(sums.argmax(dim=1).item())
            baseline = sums[0, class_idx]

            drops = torch.zeros(C, device=acts.device)

            for start in range(0, C, self.channel_batch_size):
                end = min(start + self.channel_batch_size, C)
                G = end - start

                # mask[i, ch] = 0 if ch is ablated in this micro-batch i, else 1
                mask = torch.ones(G, C, device=acts.device)
                for i in range(G):
                    mask[i, start + i] = 0.0

                input_rep = x.repeat(1, G, 1, 1, 1)  # replicate along batch

                if self.use_pre_hook:
                    def ablate_hook(module, args):
                        inp = args[0]
                        # inp: (T,G,C,H,W) or (G,C,H,W)
                        if inp.dim() == 5:
                            m = mask.view(1, G, C, 1, 1)
                        elif inp.dim() == 4:
                            m = mask.view(G, C, 1, 1)
                        else:
                            raise RuntimeError(f"Unexpected input dim: {inp.shape}")
                        return (inp * m, )
                    handle = self.target_layer.register_forward_pre_hook(ablate_hook)
                else:
                    def ablate_hook(module, inp, out):
                        # out: (T,G,C,H,W) or (G,C,H,W)
                        if out.dim() == 5:
                            m = mask.view(1, G, C, 1, 1)
                        elif out.dim() == 4:
                            m = mask.view(G, C, 1, 1)
                        else:
                            raise RuntimeError(f"Unexpected output dim: {out.shape}")
                        out.mul_(m)
                        
                    handle = self.target_layer.register_forward_hook(ablate_hook)

                try:
                    out_rep = self.model(input_rep)  # (T,G,K)
                finally:
                    handle.remove()
                    
                functional.reset_net(self.model)

                score_batch = out_rep.sum(dim=0)[:, class_idx] # (G,)
                drops[start:end] = baseline - score_batch

            self._importance = drops  # (C,)
            return drops

    def _compute_raw(self, x: torch.Tensor, class_idx: int) -> torch.Tensor:
        if self._importance is None:
            self.compute_importance(x, class_idx)
        acts = self._activations  # (T,B,C,Hf,Wf)
        imp = self._importance.view(1, 1, -1, 1, 1)
        weighted = acts * imp
        cams = weighted.sum(dim=2)      # (T,B,Hf,Wf)
        cams = cams.permute(1, 0, 2, 3) # (B,T,Hf,Wf)
        cams = F.relu(cams)
        return cams