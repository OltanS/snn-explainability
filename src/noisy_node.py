import torch
import numpy as np
from contextlib import contextmanager

from spikingjelly.activation_based import functional
from spikingjelly.activation_based.surrogate import ATan, Sigmoid
from spikingjelly.activation_based.neuron import LIFNode, IFNode

NOISE_TYPES = ["uniform", "gaussian", "atan", "sigmoid", "surrogate"]

class NoisyNodeMixin:
    def __init__(self, noise_type, **kwargs):
        super().__init__(**kwargs)
        if noise_type not in NOISE_TYPES:
            raise ValueError(f"Unsupported noise type: {noise_type}. Supported types are: {NOISE_TYPES}")
        
        if noise_type == "uniform":
            self._noise_fn = self._create_uniform_noise
        
        elif noise_type == "gaussian":
            self._noise_fn = self._create_gaussian_noise

        # force atan noise type
        elif noise_type == "atan":
            self._atan_scale = 2 / (np.pi * self.surrogate_function.alpha)
            self._noise_fn = self._create_atan_noise

        # force sigmoid noise type
        elif noise_type == "sigmoid":
            self._sigmoid_scale = 1 / self.surrogate_function.alpha
            self._noise_fn = self._create_sigmoid_noise

        # infer noise type from surrogate function
        elif noise_type == "surrogate":
            if isinstance(self.surrogate_function, ATan):
                self._atan_scale = 2 / (np.pi * self.surrogate_function.alpha)
                self._noise_fn = self._create_atan_noise

            elif isinstance(self.surrogate_function, Sigmoid):
                self._sigmoid_scale = 1 / self.surrogate_function.alpha
                self._noise_fn = self._create_sigmoid_noise
            
            else:
                raise ValueError(f"Unsupported surrogate function for surrogate noise: {type(self.surrogate_function)}")

    def _create_uniform_noise(self, v):
        return torch.rand_like(v)

    def _create_atan_noise(self, v):
        return self._atan_scale * torch.tan(np.pi * (torch.rand_like(v) - 0.5))
    
    def _create_gaussian_noise(self, v):
        return torch.randn_like(v)

    def _create_sigmoid_noise(self, v):
        u = torch.rand_like(v)
        eps = 1e-6
        u = torch.clamp(u, eps, 1 - eps)
        return self._sigmoid_scale * torch.logit(u)

    def neuronal_fire(self):
        v = self.v
        noisy_v = v + self._noise_fn(v)
        return self.surrogate_function(noisy_v - self.v_threshold)

class NoisyLIFNode(NoisyNodeMixin, LIFNode):
    pass

class NoisyIFNode(NoisyNodeMixin, IFNode):
    pass

def create_noisy_clone(original_node, noise_type):
        """
        Creates a Noisy version of the node, copying relevant hyperparameters.
        Adjust the arguments below based on your specific NoisyNode implementation.
        """
        # Common attributes for SpikingJelly nodes
        kwargs = {
            'v_threshold': original_node.v_threshold,
            'v_reset': original_node.v_reset,
            'detach_reset': original_node.detach_reset,
            'surrogate_function': original_node.surrogate_function,
            'step_mode': original_node.step_mode,
            'detach_reset' : original_node.detach_reset
        }
        if hasattr(original_node, 'tau'): kwargs['tau'] = original_node.tau
        if hasattr(original_node, 'decay_input'): kwargs['decay_input'] = original_node.decay_input

        if isinstance(original_node, IFNode):
            return NoisyIFNode(noise_type=noise_type, **kwargs)

        if isinstance(original_node, LIFNode):
            return NoisyLIFNode(noise_type=noise_type, **kwargs)

        raise ValueError(f"Unsupported node type for noise injection: {type(original_node)}")


@contextmanager
def noise_injection(model, noise_type, target_layer=None):
    """
    Context manager to temporarily replace LIF/IF nodes with Noisy versions.
    """
    _original_nodes = {} # Store: { (parent_module, child_name): original_node }
    _target_layer_backup = None
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, (LIFNode, IFNode)):
                noisy_node = create_noisy_clone(child, noise_type=noise_type)
                if noisy_node is not None:
                    # Save original for restoration
                    _original_nodes[(module, child_name)] = child
                    # Perform the swap in the model graph
                    setattr(module, child_name, noisy_node)
                    # CRITICAL: If this node is the target layer, update the reference
                    if target_layer is not None and child is target_layer:
                        _target_layer_backup = child
                        target_layer = noisy_node
    try:
        # 2. Yield control back to the computation
        yield 
    finally:
        # 3. Restore (Guaranteed execution)
        for (parent, name), original in _original_nodes.items():
            setattr(parent, name, original)
        
        # Restore the target layer reference
        if _target_layer_backup is not None:
            target_layer = _target_layer_backup
            _target_layer_backup = None
        # Clear storage
        _original_nodes = {}
        # Reset network state just in case noisy nodes left garbage
        functional.reset_net(model)