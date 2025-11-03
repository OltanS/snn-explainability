import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from spikingjelly.activation_based import neuron, surrogate, functional

class FractionalAblationCAM:
    '''
    Fractional Ablation CAM for SNNs.

    Steps:
      1. Run baseline forward; record target-layer activations over time and class score.
      2. For a set of retention fractions f in (0,1], partially zero (scale) each channel.
      3. Measure drop in target class score vs baseline for each (channel, f).
      4. Integrate drop curve over fractions (AUC) -> channel importance.
      5. Produce CAM by weighting (time-collapsed) activation maps with importance scores.
    
    Parallelization (batch size 1 assumption):
      - Channels processed in blocks of size channel_batch_size.
      - For a given fraction, we build a (G, T, C, H, W) tensor (G=block size) where each slice
        only modifies one channel, and replicate the original input to shape (G, ...).
    '''

    def __init__(self, model: nn.Module, target_layer: nn.Module, n_steps: int, channel_batch_size: int = 8):
        self.snn_model = model
        self.target_layer = target_layer
        self.n_ablation_steps = n_steps
        # Example: n_steps=4 -> tensor([0.00, 0.25, 0.50, 0.75, 1.00])
        self.fractions = torch.linspace(0.0, 1.0, n_steps + 1)  # tensor, not list (so .to(...) works later)
        self.channel_batch_size = channel_batch_size
        self.snn_model.eval()

        # alternative hook
        self._hook_into_lif = True

    def extract_activations_outputs(self, input_tensor):
        '''
        Assumes multi-step mode (T, B, C, H, W).
        Captures the pre-activation to the LIF (i.e., Conv/BN output): (T, B, C, H, W).
        '''
        activations = None
        if self._hook_into_lif:
            def pre_hook(module, input):
                x = input[0]
                nonlocal activations
                activations = x.detach().clone()
                return None
            handle = self.target_layer.register_forward_pre_hook(pre_hook)
        else:
            def hook_fn(module, input, output):
                # output: (B, C, H, W) at a single time step
                nonlocal activations
                activations = output.detach().clone()

            handle = self.target_layer.register_forward_hook(hook_fn)

        outputs = self.snn_model(input_tensor) # (B, T, num_classes)
        functional.reset_net(self.snn_model)
        handle.remove()

        return activations, outputs  # (B, T, C, H, W), (B, T, num_classes)
    
    def _forward_with_channel_scaling(self, input_tensor, scaling_vectors):
        """
        scaling_vectors: (B_rep, C) per-sample channel scaling factors.
        Applies the same scaling at every time step the target layer is invoked.
        """
        B_rep, C = scaling_vectors.shape
        if self._hook_into_lif:
            def scaling_pre_hook(module, input):
                # input: (T, B_rep, C, H, W)
                x = input[0]
                x_scaled = x * scaling_vectors.view(1, B_rep, C, 1, 1)
                rest = tuple(input[1:] if len(input) > 1 else ())
                return (x_scaled, *rest)
            
            handle = self.target_layer.register_forward_pre_hook(scaling_pre_hook)
            
        else:
            def scaling_hook(module, inp, out):
                # out: (T, B_rep, C, H, W)
                B_rep, C = scaling_vectors.shape
                return out * scaling_vectors.view(1, B_rep, C, 1, 1)

            handle = self.target_layer.register_forward_hook(scaling_hook)
        
        outputs = self.snn_model(input_tensor)  # (T, B_rep, num_classes)
        functional.reset_net(self.snn_model)
        handle.remove()
        return outputs
    
    def compute_channel_importance(self, input_tensor, class_idx=None):
        """
        input_tensor: shape (T, 1, C, H, W)
        Returns importance per channel: (C,)
        """
        with torch.no_grad():
            activations, baseline_outputs = self.extract_activations_outputs(input_tensor)
            self.activations = activations  # store for later use
            T, B, Cc, Hh, Ww = activations.shape
            assert B == 1, "This implementation assumes batch size 1 when computing importances."
            baseline_mean = baseline_outputs.mean(dim=0)
            if class_idx is None:
                class_idx = int(torch.argmax(baseline_mean, dim=1).item())
            baseline_score = baseline_mean[0, class_idx]

            fractions = self.fractions.to(activations.device)
            Fcount = fractions.shape[0]
            drops_per_channel = torch.zeros(Cc, Fcount, device=activations.device)

            # We skip the last fraction (=1.0) since drop there is zero.
            outer_tqdm = tqdm(enumerate(fractions[:-1]), total=Fcount - 1,
                              desc="Ablation fractions", position=0)
            
            for f_idx, frac in outer_tqdm:
                for start in range(0, Cc, self.channel_batch_size):
                    end = min(start + self.channel_batch_size, Cc)
                    block_channels = list(range(start, end))
                    G = end - start

                    # Build scaling matrix: start with ones, then scale one channel per sample.
                    scaling = torch.ones(G, Cc, device=activations.device)
                    for bi, ch in enumerate(block_channels):
                        scaling[bi, ch] = frac  # retention fraction

                    # Expand input for G variants
                    input_rep = input_tensor.repeat(1, G, 1, 1, 1)  # (T, G, C, H, W)

                    # Forward with channel scaling
                    ablated_outputs = self._forward_with_channel_scaling(input_rep, scaling)
                    ablated_scores = ablated_outputs.mean(dim=0)[:, class_idx]  # (G,)

                    drops = baseline_score - ablated_scores
                    drops_per_channel[start:end, f_idx] = drops

            importance = torch.trapz(drops_per_channel, fractions, dim=1)
            return importance, activations, drops_per_channel, fractions
        
    def rescale_cams(self, cams, input_tensor):
        B, T, Cc, Hf, Wf = cams.shape
        H0, W0 = input_tensor.shape[-2:]
        # reshape to (B*T, Cc, H', W') for interpolation
        cams = cams.view(B * T, Cc, Hf, Wf)
        # bilinear upsample to original H0×W0
        cams = F.interpolate(cams, size=(H0, W0), mode='bilinear', align_corners=False)
        # back to (B, T, Cc, H0, W0)
        cams = cams.view(B, T, Cc, H0, W0)
        return cams

    def compute_cam(self, input_tensor, importance, normalize_per_timestep=False, apply_relu=True):
        """
        After getting the channel importance scores we create per-timestep CAMs.
        input_tensor: (T, 1, C, H, W)
        Returns cams: (T, H, W)
        """
        activations = self.activations  # (T, 1, C, H, W)
        assert activations.dim() == 5 and activations.size(1) == 1
        T, _, C, H, W = activations.shape

        a = activations[:, 0]                                # (T, C, H, W)
        weighted = a * importance.view(1, C, 1, 1)           # (T, C, H, W)
        cams = weighted.sum(dim=1)                           # (T, H, W)
        
        if apply_relu:
            cams = F.relu(cams)

        if normalize_per_timestep:
            cams_flat = cams.view(T, -1)
            mins = cams_flat.min(dim=1).values.view(T, 1, 1)
            maxs = cams_flat.max(dim=1).values.view(T, 1, 1)
            denom = (maxs - mins).clamp_min(1e-8)
            cams = (cams - mins) / denom

        upscaled_cams = self.rescale_cams(cams.unsqueeze(0).unsqueeze(2), input_tensor)  # (1, T, 1, H_in, W_in)
        
        return upscaled_cams[0, :, 0, :, :]  # (T, H_in, W_in)


            
            
                    