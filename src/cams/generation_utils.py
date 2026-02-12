import torch
from spikingjelly.activation_based import functional
from src.cams.gradcam import SNNGradCAM
from src.cams.ablation_cam import AblationCAM
from src.cams.mc_ablation_cam import MCAblationCAM
from src.cams.sam import SpikeActivationMap
from src.cams.guided_backprop import GuidedBackpropCAM
from src.cams.mc_ablation_cam_transient import MCAblationCAMTransient
from src.cams.noise_gradcam import SNNNoiseGradCAM
from src.cams.replace_surrogate_gradcam import ReplaceSurrogateGradCAM
from src.cams.smoothgrad_cam import SNNSmoothGradCAM
from src.cams.smoothnoisegrad_cam import SNNSmoothNoiseGradCAM
from src.cams.grad3d import SNNGrad3D
from src.cams.integrated_gradients import SNNIntegratedGradients

from src.cams.cam_utils import grad_cam_context
from src.utils.model_utils import get_conv_layers, get_lif_layers

def generate_grad_cam(events, class_idx, target_layer_idx, model, hook_to_conv=True):
    with grad_cam_context(model):
        target = get_conv_layers(model)[target_layer_idx] if hook_to_conv else get_lif_layers(model)[target_layer_idx]
        grad_cam = SNNGradCAM(model, target)
        return grad_cam.generate(events, class_idx=class_idx).detach()

def generate_smooth_grad_cam(events, class_idx, target_layer_idx, model, n_samples=25, stdev_spread=0.15, hook_to_conv=True):
    with grad_cam_context(model):
        target = get_conv_layers(model)[target_layer_idx] if hook_to_conv else get_lif_layers(model)[target_layer_idx]
        smooth_grad_cam = SNNSmoothGradCAM(model, target, n_samples=n_samples, stdev_spread=stdev_spread)
        return smooth_grad_cam.generate(events, class_idx=class_idx).detach()

def generate_noise_grad_cam(events, class_idx, target_layer_idx, model, k_trials=100, batch_size=16, hook_to_conv=True, noise_type="surrogate"):
    with grad_cam_context(model):
        target = get_conv_layers(model)[target_layer_idx] if hook_to_conv else get_lif_layers(model)[target_layer_idx]
        noise_grad_cam = SNNNoiseGradCAM(model, target, k_trials=k_trials, batch_size=batch_size, noise_type=noise_type)
        return noise_grad_cam.generate(events, class_idx=class_idx).detach()

def generate_smooth_noise_grad_cam(events, class_idx, target_layer_idx, model, k_trials=100, batch_size=16, stdev_spread=0.15, hook_to_conv=True, noise_type="gaussian"):
    """
    Generates CAM using Doubly Stochastic approach: Input Noise + Internal Noise.
    """
    with grad_cam_context(model):
        target = get_conv_layers(model)[target_layer_idx] if hook_to_conv else get_lif_layers(model)[target_layer_idx]
        cam = SNNSmoothNoiseGradCAM(
            model, 
            target, 
            k_trials=k_trials, 
            batch_size=batch_size, 
            stdev_spread=stdev_spread, 
            noise_type=noise_type
        )
        return cam.generate(events, class_idx=class_idx).detach()

def generate_replace_surrogate_grad_cam(events, class_idx, target_layer_idx, model, new_surrogate=None):
    with grad_cam_context(model):
        target = get_conv_layers(model)[target_layer_idx]
        replace_surrogate_grad_cam = ReplaceSurrogateGradCAM(model, target, new_surrogate=new_surrogate)
        return replace_surrogate_grad_cam.generate(events, class_idx=class_idx).detach()

def generate_ablation_cam(events, class_idx, target_layer_idx, model, batch_size=32, use_pre_hook=False):
    functional.reset_net(model)
    target = get_conv_layers(model)[target_layer_idx]
    ab_cam = AblationCAM(model, target, batch_size=batch_size, use_pre_hook=use_pre_hook)
    return ab_cam.generate(events, class_idx=class_idx).detach()

def generate_mc_ablation_cam(events, class_idx, target_layer_idx, model, k_trials=20, batch_size=32, use_post_hook=False):
    functional.reset_net(model)
    target = get_lif_layers(model)[target_layer_idx]
    mc_cam = MCAblationCAM(model, target, batch_size=batch_size, k_trials=k_trials, use_post_hook=use_post_hook)
    return mc_cam.generate(events, class_idx=class_idx).detach(), mc_cam

def generate_mc_ablation_cam_transient(events, class_idx, target_layer_idx, model, k_trials=20, batch_size=32, noise_type="surrogate"):
    functional.reset_net(model)
    target = get_lif_layers(model)[target_layer_idx]
    mc_cam_transient = MCAblationCAMTransient(model, target, batch_size=batch_size, k_trials=k_trials, noise_type=noise_type)
    return mc_cam_transient.generate(events, class_idx=class_idx).detach(), mc_cam_transient

def generate_sam(events, target_layer_idx, model):
    functional.reset_net(model)
    target = get_lif_layers(model)[target_layer_idx]
    sam = SpikeActivationMap(model, target)
    return sam.generate(events).detach()

def generate_guided_backprop_cam(events, class_idx, model):
    with grad_cam_context(model):
        guided_backprop_cam = GuidedBackpropCAM(model)
        return guided_backprop_cam.generate(events, class_idx=class_idx, upsample=False).detach()

def generate_grad3d(events, class_idx, model):
    """
    Compute SNN-Grad3D (saliency of class score w.r.t. input).
    Returns tensor of shape (B, T, H, W).
    """
    functional.reset_net(model)
    with grad_cam_context(model):
        grad3d = SNNGrad3D(model)
        return grad3d.generate(events, class_idx=class_idx, upsample=False).detach()

def generate_integrated_gradients(events, class_idx, model, steps=50, mode="3d", baseline=None):
    """
    Compute SNN-Integrated-Gradients. mode in {'3d','2d'}.
    Returns tensor (B, T, H, W).
    """
    functional.reset_net(model)
    with grad_cam_context(model):
        ig = SNNIntegratedGradients(model, steps=steps, mode=mode, baseline=baseline)
        return ig.generate(events, class_idx=class_idx, upsample=False).detach()