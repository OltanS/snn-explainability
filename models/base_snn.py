import torch
import torch.nn as nn

import snntorch as snn

class BaseSNN(nn.Module):
    '''
    All the functionality here can be achieved with a wrapper and hooks as well, doing it this way
    since I know it better this way!
    '''
    def __init__(self):
        super(BaseSNN, self).__init__()

    def forward(self, x):
        self.reset_memories()
        return self._forward_without_reset(x)
    
    def _forward_without_reset(self, x):
        B, T, C, H, W = x.shape # batch size, number of time bins, number of channels(polarities), height, width
        mem_histories = self.create_mem_histories()
        outputs = []
        for t in range(T):
            f = self._forward_one_step(x, t)
            outputs.append(f)
            self.collect_memories(mem_histories)
        self.format_mem_histories(mem_histories)
        outputs = torch.stack(outputs, dim=1)  # Stack outputs along the time dimension
        return outputs, mem_histories

    def _forward_one_step(self, x, t):
        """
        Perform a forward pass for one time step.
        X shape -> B, T, C, H, W
        t -> current time step index
        It should only process the input for the current time step `t`.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def create_mem_histories(self):
        """
        Create a dictionary to store membrane potential histories for each layer.
        """
        mem_histories = {}
        for name, module in self.named_modules():
            if isinstance(module, snn.Leaky):
                mem_histories[name] = []
        return mem_histories
    
    def reset_memories(self):
        """
        Reset the membrane potentials of all Leaky Integrate-and-Fire layers.
        """
        for name, module in self.named_modules():
            if isinstance(module, snn.Leaky):
                module.reset_mem()
    
    def collect_memories(self, mem_histories):
        """
        Collect the membrane potentials from each Leaky Integrate-and-Fire layer
        at each time step and store them in the provided dictionary.
        """
        for name, module in self.named_modules():
            if isinstance(module, snn.Leaky):
                mem_histories[name].append(module.mem.detach().clone())

    def format_mem_histories(self, mem_histories):
        """
        Convert the membrane potential histories to a tensor format.
        The format is Batch x Time x Channels x Height x Width.
        """
        for name in mem_histories:
            mem_histories[name] = torch.stack(mem_histories[name], dim=0).transpose(1, 0)
    
    def restore_membrane_states(self, mem_histories, timestep):
        """
        Restore the membrane states of the Leaky Integrate-and-Fire layers
        to the state at a specific time step.
        mem_histories: Dictionary containing membrane histories across batches and time.
        Restores to the given timestep INDEX. so if timestep=0, it restores to the initial state.
        """
        for name, module in self.named_modules():
            if isinstance(module, snn.Leaky) and name in mem_histories:
                module_mem = mem_histories[name][:, timestep]
                module.mem = module_mem.clone()
    
    def forward_from_state(self, x, timestep, mem_histories):
        """
        Given membrane histories and a specific timestep,
        restore the membrane states and perform the forward pass from that state.
        x -> batch_size, time_steps, channels, height, width
        """
        self.restore_membrane_states(mem_histories, timestep)
        return self._forward_without_reset(x)
        
