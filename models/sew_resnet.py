import warnings
from functools import partial

import torch
import torch.nn as nn
import snntorch as snn

from models.base_snn import BaseSNN
from models.model_utils import BatchNorm2dTTWrapper

def sew_function(x: torch.Tensor, y: torch.Tensor, cnf:str):
    if cnf == 'ADD':
        return x + y
    elif cnf == 'AND':
        return x * y
    elif cnf == 'IAND':
        return x * (1. - y)
    else:
        raise NotImplementedError


class BasicBlock(BaseSNN):
    def __init__(self, in_channels, out_channels, lif, norm, stride=1, downsample=None, cnf="ADD"):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm(input_features=out_channels)
        self.lif1 = lif()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm(input_features=out_channels)
        self.lif2 = lif()
        self.stroke = stride
        self.cnf = cnf
        self.downsample = downsample

        if downsample is not None:
            self.downsample_norm = norm(input_features=out_channels)
            self.downsample_lif = lif()

    def _forward_one_step(self, xt, t):
        identity = xt
        B, C, H, W = xt.shape
        if self.downsample is not None:
            identity = self.downsample(identity)
            identity = self.downsample_norm[t](identity)
            identity = self.downsample_lif(identity)
      
        c1 = self.conv1(xt)
        b1 = self.bn1[t](c1)
        s1 = self.lif1(b1)

        c2 = self.conv2(s1)
        b2 = self.bn2[t](c2)
        s2 = self.lif2(b2)

        out = sew_function(identity, s2, self.cnf)
        return out


class SEWResnet(BaseSNN):
    '''
    SNNtorch implementation of the SEW-ResNet 34 architecture, as described in:
    https://github.com/fangwei123456/spikingjelly/blob/master/spikingjelly/activation_based/model/sew_resnet.py
    '''
    def __init__(self, num_classes, num_channels, time_steps, beta, enable_bntt):
        super(SEWResnet, self).__init__()
        if enable_bntt:
            self.norm_partial = partial(snn.BatchNormTT2d, time_steps=time_steps)
        else:
            warnings.warn("BNTT disabled, using nn.BatchNorm instead.", UserWarning)
            self.norm_partial = partial(BatchNorm2dTTWrapper, time_steps=time_steps)
        blocks = [3, 4, 6, 3]
        self.in_channels = 64

        self.lif_partial = partial(snn.Leaky, beta=beta, init_hidden=True)

        # initial conv layer
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self.norm_partial(input_features=64)
        self.lif1 = self.lif_partial()
        # lets do an avgpool over a maxpool, since maxpools aren't so meaningful in SNNs.
        self.avgpool1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, blocks[0])
        self.layer2 = self._make_layer(128, blocks[1], stride=2)
        self.layer3 = self._make_layer(256, blocks[2], stride=2)
        self.layer4 = self._make_layer(512, blocks[3], stride=2)

        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

        self.apply(self._init_weights)

    def _make_layer(self, out_channels, blocks, stride=1, cnf="ADD"):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, self.lif_partial, self.norm_partial, stride=stride, downsample=downsample, cnf=cnf))
        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, self.lif_partial, self.norm_partial, cnf=cnf))
        
        return nn.Sequential(*layers)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            m.weight.data *= 0.5
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 0.5)  # Conservative scaling
            if m.bias is not None:
                nn.init.constant_(m.bias, -0.1)   # Slight negative bias
            
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            m.weight.data *= 0.3  # More conservative for final layers
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def _forward_one_step(self, xt, t):
        B, C, H, W = xt.shape
        out = self.conv1(xt)
        out = self.bn1[t](out)
        out = self.lif1(out)
        out = self.avgpool1(out)

        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer: # loop through sequential
                out = block._forward_one_step(out, t)
        
        out = self.avgpool2(out)
        out = out.view(B, -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
