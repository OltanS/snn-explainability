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
        return torch.randn_like(v) * 0.1

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
    Context manager to replace LIF/IF nodes with Noisy versions.
    
    If target_layer is provided:
        Only nodes temporally/topologically AFTER the target_layer will be replaced.
        (Useful for CAMs where you want gradients to flow back TO the target, 
         so you need noise in the 'gate' layers between Output and Target).
         
    If target_layer is None:
        All nodes in the network are replaced.
    """
    _original_nodes = {}
    updated_target = target_layer
    should_inject = (target_layer is None)

    # We iterate through modules to find the target and subsequent nodes.
    for module_name, module in model.named_modules():
        for child_name, child in module.named_children():
            
            # If we hit the target layer, we enable injection for all SUBSEQUENT layers.
            if target_layer is not None and child is target_layer:
                should_inject = True
            
            # 2. REPLACE IF ENABLED
            if should_inject and isinstance(child, (LIFNode, IFNode)):
                noisy_node = create_noisy_clone(child, noise_type=noise_type)
                _original_nodes[(module, child_name)] = child
                setattr(module, child_name, noisy_node)
                # Update the reference if we just replaced the target layer object
                if target_layer is not None and child is target_layer:
                    updated_target = noisy_node
    try:
        yield updated_target
    finally:
        # Restore original nodes
        for (parent, name), original in _original_nodes.items():
            setattr(parent, name, original)
        functional.reset_net(model)


def test_noisy_fire_negative_voltage():
    print("Testing NoisyIFNode voltage drop behavior over time...\n")
    
    # Create a NoisyIFNode with soft reset (v_reset=None)
    # Soft reset formula: v_new = v_old - v_threshold * spike
    node = NoisyIFNode(noise_type="uniform", v_threshold=1.0, v_reset=None)
    
    # Monkeypatch the noise function to return a fixed positive noise
    # This ensures we trigger a spike deterministically without relying on RNG.
    # We want a scenario where: v < threshold AND v + noise > threshold
    # Noise is set to 0.5
    node._noise_fn = lambda v: torch.full_like(v, 0.5)
    
    # Inputs for 3 time steps
    # Step 0: Input 0.8. V_pre=0.8. Noisy=1.3 (>1.0). Spike=1. V_post = 0.8 - 1.0 = -0.2.
    # Step 1: Input 0.8. V_pre=-0.2+0.8=0.6. Noisy=1.1 (>1.0). Spike=1. V_post = 0.6 - 1.0 = -0.4.
    # Step 2: Input 0.1. V_pre=-0.4+0.1=-0.3. Noisy=0.2 (<1.0). Spike=0. V_post = -0.3.
    inputs = torch.tensor([0.8, 0.8, 0.1])
    
    print(f"{'Time':<5} | {'Input':<10} | {'V_mem (pre)':<12} | {'Noise':<10} | {'V_noisy':<10} | {'Spike':<5} | {'V_mem (post)':<12}")
    print("-" * 85)

    for t, x in enumerate(inputs):
        x_in = x.unsqueeze(0) # Batch dim
        
        # Manual forward pass steps to inspect intermediates
        node.neuronal_charge(x_in)
        v_pre = node.v.item()
        
        # Replicate noise calculation for display (since we monkeypatched it, we know it's 0.5)
        noise_val = node._noise_fn(node.v).item()
        v_noisy = v_pre + noise_val
        
        spike = node.neuronal_fire()
        
        node.neuronal_reset(spike)
        v_post = node.v.item()
        
        print(f"{t:<5} | {x.item():<10.4f} | {v_pre:<12.4f} | {noise_val:<10.4f} | {v_noisy:<10.4f} | {int(spike.item()):<5} | {v_post:<12.4f}")

    # Assertions
    assert node.v.item() < 0.0, f"Final voltage expected < 0, but got {node.v.item()}"
    print("\nTest Passed: Voltage dropped below 0 due to noise-induced spike with soft reset.") 

if __name__ == "__main__":
    test_noisy_fire_negative_voltage()