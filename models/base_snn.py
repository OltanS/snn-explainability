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
        '''
        returns -> outputs and mem_histories. mem_histories[t] is the membrane potential before the forward pass at time t + after the last forward pass. so there are T+1 entries in the list.
        '''
        self.reset_memories()
        return self._forward_without_reset(x)
    
    def _forward_without_reset(self, x):
        B, T, C, H, W = x.shape # batch size, number of time bins, number of channels(polarities), height, width
        outputs = []
        for t in range(T):
            xt = x[:,t]
            f = self._forward_one_step(xt, t)
            outputs.append(f)

        outputs = torch.stack(outputs, dim=1)  # Stack outputs along the time dimension
        return outputs

    def _forward_one_step(self, xt, t):
        """
        Perform a forward pass for one time step.
        X shape -> B, C, H, W
        t -> current time step index
        It should only process the input for the current time step `t`.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def reset_memories(self):
        """
        Reset the membrane potentials of all Leaky Integrate-and-Fire layers.
        """
        for name, module in self.named_modules():
            if isinstance(module, snn.Leaky):
                module.reset_mem()
