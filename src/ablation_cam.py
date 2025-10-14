import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from models.base_snn import BaseSNN

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

    def __init__(self, model: BaseSNN, target_layer: nn.Module, n_steps: int, channel_batch_size: int = 8):
        self.snn_model = model
        self.target_layer = target_layer
        self.n_ablation_steps = n_steps
        # Sample points now cover the closed interval [0, 1], adding the full-retention endpoint.
        # We keep n_steps as "number of internal intervals", giving n_steps+1 fraction points.
        # Example: n_steps=4 -> tensor([0.00, 0.25, 0.50, 0.75, 1.00])
        self.fractions = torch.linspace(0.0, 1.0, n_steps + 1)  # tensor, not list (so .to(...) works later)
            # This controls how many (channel, fraction) pairs to process in parallel
        # e.g., channel_batch_size=8 means we process 8 ablation combinations simultaneously
        self.channel_batch_size = channel_batch_size
        self.snn_model.eval()

    def extract_activations_outputs(self, input_tensor, target_layer):
        activations = []

        def hook_fn(module, input, output):
            # output: (B, C, H, W) at a single time step
            activations.append(output.detach().clone())

        handle = target_layer.register_forward_hook(hook_fn)
        outputs, _ = self.snn_model(input_tensor) # (B, T, num_classes)
        handle.remove()

        return torch.stack(activations, dim=1), outputs  # Stack along time dimension, (B, T, C, H, W), (B, T, num_classes)
    
    def _forward_with_channel_scaling(self, input_tensor, scaling_vectors):
        """
        scaling_vectors: (B_rep, C) per-sample channel scaling factors.
        Applies the same scaling at every time step the target layer is invoked.
        """
        def scaling_hook(module, inp, out):
            # out: (B_rep, C, H, W)
            return out * scaling_vectors[:, :, None, None]

        handle = self.target_layer.register_forward_hook(scaling_hook)
        outputs, _ = self.snn_model(input_tensor)  # (B_rep, T, num_classes)
        handle.remove()
        return outputs
    
    def compute_channel_importance(self, input_tensor, class_idx=None):
        """
        input_tensor: shape (1, T_in, C_in, H_in, W_in)
        """
        with torch.no_grad():
            activations, baseline_outputs = self.extract_activations_outputs(input_tensor, self.target_layer)
            B, T, Cc, Hh, Ww = activations.shape
            assert B == 1
            baseline_mean = baseline_outputs.mean(dim=1)
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
                inner_tqdm = tqdm(range(0, Cc, self.channel_batch_size),
                                  desc="Channel blocks", position=1, leave=False)
                for start in inner_tqdm:
                    end = min(start + self.channel_batch_size, Cc)
                    block_channels = list(range(start, end))
                    G = end - start

                    # Build scaling matrix: start with ones, then scale one channel per sample.
                    scaling = torch.ones(G, Cc, device=activations.device)
                    for bi, ch in enumerate(block_channels):
                        scaling[bi, ch] = frac  # retention fraction

                    # Expand input for G variants
                    input_rep = input_tensor.expand(G, *input_tensor.shape[1:])

                    # Forward with channel scaling
                    ablated_outputs = self._forward_with_channel_scaling(input_rep, scaling)
                    ablated_scores = ablated_outputs.mean(dim=1)[:, class_idx]  # (G,)

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

    def compute_cam(self, input_tensor, activations, importance, normalize_per_timestep=False, apply_relu=True):
        """
        Goal:
        After getting the channel importance scores we want to create per-timestep CAMs.
        input_tensor: (1, T_in, C_in, H_in, W_in) or model-specific.
        For each timestep, we scale the activation maps by the importance scores. We then upscale the CAMs to the input size.
        normalize_per_timestep: if True, normalize each timestep's CAM to [0, 1] individually; else global min-max.
        Returns:
        cams: (T, H_in, W_in)
        """
        assert activations.dim() == 5 and activations.size(0) == 1
        _, T, C, H, W = activations.shape

        # imp_min, imp_max = importance.min(), importance.max()
        # if (imp_max - imp_min) > 1e-8:
        #     importance_norm = (importance - imp_min) / (imp_max - imp_min + 1e-8)
        # else:
        #     importance_norm = torch.zeros_like(importance)
        # print('Importance scores min/max:', imp_min.item(), imp_max.item())
        # print('Importances after normalization min/max:', importance_norm.min().item(), importance_norm.max().item())
        
        # multiply the activations by the normalized importance scores
        weighted = activations[0] * importance.view(1, C, 1, 1)  # (T, C, H, W)
        cams = weighted.sum(dim=1)  # (T, H, W)
        
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


            
            
                    