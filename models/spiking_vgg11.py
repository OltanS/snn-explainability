import torch
import torch.nn as nn

import snntorch as snn
from snntorch import utils


class BatchNormTT2dFixed(nn.Module):
    '''Original is missing forward function for modulelist'''

    def __init__(self, input_features, time_steps):
        super(BatchNormTT2dFixed, self).__init__()
        self.batchnorm = snn.BatchNormTT2d(input_features, time_steps=time_steps)

    def forward(self, x):
        for layer in self.batchnorm:
            x = layer(x)
        return x
    
    def reset(self):
        # Reset each bn module as needed.
        for bn in self.batchnorm:
            if hasattr(bn, "reset"):
                bn.reset()        

class BatchNormTT1dFixed(nn.Module):
    '''Original is missing forward function for modulelist'''

    def __init__(self, input_features, time_steps):
        super(BatchNormTT1dFixed, self).__init__()
        self.batchnorm = snn.BatchNormTT1d(input_features, time_steps=time_steps)

    def forward(self, x):
        for layer in self.batchnorm:
            x = layer(x)
        return x
    
    def reset(self):
        # Reset each bn module as needed.
        for bn in self.batchnorm:
            if hasattr(bn, "reset"):
                bn.reset()

class SpikingVGG11(nn.Module):
    '''
    Implemented in line the network described in the supplementary material of:
    Kim, Y., Panda, P., 2021. Visual explanations from spiking neural networks using inter-spike intervals. Sci Rep 11, 19037. https://doi.org/10.1038/s41598-021-98448-0

    It is designed around tiny-imagenet, as was used in the paper mentioned above.
    Note that it originally outputted logits to be softmaxed, not spikes. We elect to stick to the spike-based nature and add a Leaky layer at the end.
    '''
    def __init__(self, num_classes=200, time_steps=30, beta=0.95):
        super(SpikingVGG11, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.time_steps = time_steps
        
        self.conv_blocks = nn.ModuleList()
        self.conv_blocks.append(self.make_conv_block(3, 64, time_steps, beta)) # 1
        self.conv_blocks.append(self.make_conv_block(64, 128, time_steps, beta)) # 2
        self.conv_blocks.append(self.make_conv_block(128, 256, time_steps, beta)) # 3
        self.conv_blocks.append(self.make_conv_block(256, 256, time_steps, beta)) # 4
        self.conv_blocks.append(self.make_conv_block(256, 512, time_steps, beta)) # 5
        self.conv_blocks.append(self.make_conv_block(512, 512, time_steps, beta)) # 6
        self.conv_blocks.append(self.make_conv_block(512, 512, time_steps, beta)) # 7
        self.conv_blocks.append(self.make_conv_block(512, 512, time_steps, beta)) # 8
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 4096),
            BatchNormTT1dFixed(4096, time_steps=time_steps),
            snn.Leaky(beta=beta, init_hidden=True),
            nn.Linear(4096, num_classes),
            BatchNormTT1dFixed(num_classes, time_steps=time_steps),
            snn.Leaky(beta=beta)
        )

    def make_conv_block(self, in_channels, out_channels, time_steps, beta):
        '''
        Conv2d -> BatchNormTT2d -> Leaky
        '''
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            BatchNormTT2dFixed(out_channels, time_steps=time_steps),
            snn.Leaky(beta=beta, init_hidden=True)
        )
    
    def forward(self, x):
        for block in self.conv_blocks:
            utils.reset(block)
        utils.reset(self.classifier)
        B, T, C, H, W = x.shape  # batch size, time steps, channels, height, width

        out_spikes = []
        out_mem = []
        for t in range(self.time_steps):
            xt = x[:, t]
            xt = self.conv_blocks[0](xt)
            xt = self.avgpool(xt)
            xt = self.conv_blocks[1](xt)
            xt = self.avgpool(xt)
            xt = self.conv_blocks[2](xt)
            xt = self.conv_blocks[3](xt)
            xt = self.avgpool(xt)
            xt = self.conv_blocks[4](xt)
            xt = self.conv_blocks[5](xt)
            xt = self.avgpool(xt)
            xt = self.conv_blocks[6](xt)
            xt = self.conv_blocks[7](xt)
            xt = self.avgpool(xt)
            xt = xt.view(B, -1)
            out_spike, mem_t = self.classifier(xt)
            out_spikes.append(out_spike)
            out_mem.append(mem_t)

        out_spikes = torch.stack(out_spikes, dim=1)  # B, T, num_classes
        out_mem = torch.stack(out_mem, dim=1)

        return out_spikes, out_mem  # B, T, num_classes
