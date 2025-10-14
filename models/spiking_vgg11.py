import warnings
from functools import partial

import torch
import torch.nn as nn
import snntorch as snn

from models.base_snn import BaseSNN
from models.model_utils import BatchNorm2dTTWrapper, BatchNorm1dTTWrapper

class SpikingVGG11(BaseSNN):
    def __init__(self, num_classes, num_channels, time_steps, beta, enable_bntt):
        super(SpikingVGG11, self).__init__()
        
        # choose 2d and 1d batchnorm factories
        if enable_bntt:
            self.norm2d_partial = partial(snn.BatchNormTT2d, time_steps=time_steps)
            self.norm1d_partial = partial(snn.BatchNormTT1d, time_steps=time_steps)
        else:
            warnings.warn("BNTT disabled, using nn.BatchNorm instead.", UserWarning)
            self.norm2d_partial = partial(BatchNorm2dTTWrapper, time_steps=time_steps)
            self.norm1d_partial = partial(BatchNorm1dTTWrapper, time_steps=time_steps)

        self.time_steps = time_steps
        affine_flag = True
        bias_flag   = False

        # conv + bn layers
        self.conv1 = nn.Conv2d(num_channels,  64,  kernel_size=3, padding=1, bias=bias_flag)
        self.bn1   = self.norm2d_partial(input_features=64,  eps=1e-4, affine=affine_flag)
        self.lif1  = snn.Leaky(beta=beta, init_hidden=True)

        self.conv2 = nn.Conv2d(64,             128, kernel_size=3, padding=1, bias=bias_flag)
        self.bn2   = self.norm2d_partial(input_features=128, eps=1e-4, affine=affine_flag)
        self.lif2  = snn.Leaky(beta=beta, init_hidden=True)

        self.conv3 = nn.Conv2d(128,            256, kernel_size=3, padding=1, bias=bias_flag)
        self.bn3   = self.norm2d_partial(input_features=256, eps=1e-4, affine=affine_flag)
        self.lif3  = snn.Leaky(beta=beta, init_hidden=True)
        self.conv4 = nn.Conv2d(256,            256, kernel_size=3, padding=1, bias=bias_flag)
        self.bn4   = self.norm2d_partial(input_features=256, eps=1e-4, affine=affine_flag)
        self.lif4  = snn.Leaky(beta=beta, init_hidden=True)

        self.conv5 = nn.Conv2d(256,            512, kernel_size=3, padding=1, bias=bias_flag)
        self.bn5   = self.norm2d_partial(input_features=512, eps=1e-4, affine=affine_flag)
        self.lif5  = snn.Leaky(beta=beta, init_hidden=True)
        self.conv6 = nn.Conv2d(512,            512, kernel_size=3, padding=1, bias=bias_flag)
        self.bn6   = self.norm2d_partial(input_features=512, eps=1e-4, affine=affine_flag)
        self.lif6  = snn.Leaky(beta=beta, init_hidden=True)

        self.conv7 = nn.Conv2d(512,            512, kernel_size=3, padding=1, bias=bias_flag)
        self.bn7   = self.norm2d_partial(input_features=512, eps=1e-4, affine=affine_flag)
        self.lif7  = snn.Leaky(beta=beta, init_hidden=True)
        self.conv8 = nn.Conv2d(512,            512, kernel_size=3, padding=1, bias=bias_flag)
        self.bn8   = self.norm2d_partial(input_features=512, eps=1e-4, affine=affine_flag)
        self.lif8  = snn.Leaky(beta=beta, init_hidden=True)

        # classifier
        self.fc1     = nn.Linear(512, 4096, bias=bias_flag)
        self.bn9     = self.norm1d_partial(input_features=4096, eps=1e-4, affine=affine_flag)
        self.lif9    = snn.Leaky(beta=beta, init_hidden=True)
        self.fc2     = nn.Linear(4096, num_classes, bias=False)

        # pooling / dropout
        self.pool          = nn.AvgPool2d(2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1,1))
        # self.fc_dropout    = nn.Dropout(p=0.5)

        # self.lif10 = snn.Leaky(beta=beta, init_hidden=True)
        self.apply(self._init_weights)
    
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
        c1 = self.conv1(xt)
        b1 = self.bn1[t](c1)
        s1 = self.lif1(b1)
        s1 = self.pool(s1)

        c2 = self.conv2(s1)
        b2 = self.bn2[t](c2)
        s2 = self.lif2(b2)
        s2 = self.pool(s2)

        c3 = self.conv3(s2)
        b3 = self.bn3[t](c3)
        s3 = self.lif3(b3)
        c4 = self.conv4(s3)
        b4 = self.bn4[t](c4)
        s4 = self.lif4(b4)
        s4 = self.pool(s4)

        c5 = self.conv5(s4)
        b5 = self.bn5[t](c5)
        s5 = self.lif5(b5)
        c6 = self.conv6(s5)
        b6 = self.bn6[t](c6)
        s6 = self.lif6(b6)
        s6 = self.pool(s6)

        c7 = self.conv7(s6)
        b7 = self.bn7[t](c7)
        s7 = self.lif7(b7)
        c8 = self.conv8(s7)
        b8 = self.bn8[t](c8)
        s8 = self.lif8(b8)
        s8 = self.pool(s8)

        g2 = self.adaptive_pool(s8).view(B, -1)
        f1 = self.fc1(g2)
        b9 = self.bn9[t](f1)
        s9 = self.lif9(b9)
        # s9 = self.fc_dropout(s9)
        f2 = self.fc2(s9)

        # f2 = self.lif10(f2)

        return f2
