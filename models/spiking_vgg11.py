import torch
import torch.nn as nn

import snntorch as snn
from snntorch import utils

class SpikingConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_steps, beta, enable_bntt):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn   = snn.BatchNormTT2d(out_ch, time_steps) if enable_bntt else nn.ModuleList([nn.Identity() for _ in range(time_steps)])
        self.lif  = snn.Leaky(beta=beta, init_hidden=True)
    def forward(self, x, t):
        x = self.conv(x)
        x = self.bn[t](x)
        x = self.lif(x)
        return x

    def reset(self):
        self.lif.reset_hidden()
        self.lif.detach_hidden()

class SpikingClassifier(nn.Module):
    def __init__(self, num_classes, time_steps, beta, enable_bntt):
        super().__init__()
        self.fc1  = nn.Linear(512 * 2 * 2, 4096)
        self.bn1  = snn.BatchNormTT1d(4096, time_steps) if enable_bntt else nn.ModuleList([nn.Identity() for _ in range(time_steps)])
        self.lif1 = snn.Leaky(beta=beta, init_hidden=True)
        self.fc2  = nn.Linear(4096, num_classes)

    def forward(self, x, t):
        x = self.fc1(x)
        x = self.bn1[t](x)
        x = self.lif1(x)
        x = self.fc2(x)
        return x

    def reset(self):
        self.lif1.reset_hidden()
        self.lif1.detach_hidden()

class SpikingVGG11(nn.Module):
    '''
    Implemented in line the network described in the supplementary material of:
    Kim, Y., Panda, P., 2021. Visual explanations from spiking neural networks using inter-spike intervals. Sci Rep 11, 19037. https://doi.org/10.1038/s41598-021-98448-0

    It is designed around tiny-imagenet as was used in the paper mentioned above, with an adaptive average pool before the classifier to be compatible with larger image sizes.
    '''
    def __init__(self, num_classes, num_channels, time_steps, beta, enable_bntt):
        super(SpikingVGG11, self).__init__()
        self.time_steps = time_steps
        # vgg‐style config
        cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512]
        self.features = nn.ModuleList()
        c = num_channels
        for v in cfg:
            if v == 'M':
                self.features.append(nn.AvgPool2d(2,2))
            else:
                self.features.append(SpikingConvBlock(c, v, time_steps, beta, enable_bntt))
                c = v
  
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2,2))
        self.classifier = SpikingClassifier(num_classes, time_steps, beta, enable_bntt)
    
    def reset_hidden(self):
        for layer in self.features:
            if isinstance(layer, SpikingConvBlock):
                layer.reset()
        
        self.classifier.reset()

    def forward(self, x):
        self.reset_hidden()
        B, T, C, H, W = x.shape
        out = []
        for t in range(T):
            xt = x[:,t]
            for layer in self.features:
                # pooling layers only take x, our blocks take (x,t)
                xt = layer(xt, t) if isinstance(layer, (SpikingConvBlock,)) else layer(xt)
            xt = self.adaptive_pool(xt)
            xt = xt.view(B, -1)
            xt = self.classifier(xt, t)
            out.append(xt)
        return torch.stack(out, dim=1)
