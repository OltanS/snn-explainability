import torch
import torch.nn as nn

import snntorch as snn

from models.base_snn import BaseSNN

class SpikingVGG11(BaseSNN):
    '''
    Implemented in line the network described in the supplementary material of:
    Kim, Y., Panda, P., 2021. Visual explanations from spiking neural networks using inter-spike intervals. Sci Rep 11, 19037. https://doi.org/10.1038/s41598-021-98448-0

    It is designed around tiny-imagenet as was used in the paper mentioned above, with an adaptive average pool before the classifier to be compatible with larger image sizes.

    Re-written in a super simplex way since I can't seem to get it work with more modular approaches!
    '''
    def __init__(self, num_classes, num_channels, time_steps, beta, enable_bntt):
        super(SpikingVGG11, self).__init__()
        self.time_steps = time_steps
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1   = snn.BatchNormTT2d(64, time_steps) if enable_bntt else nn.Identity()
        self.lif1 = snn.Leaky(beta=beta, init_hidden=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2   = snn.BatchNormTT2d(128, time_steps) if enable_bntt else nn.Identity()
        self.lif2 = snn.Leaky(beta=beta, init_hidden=True)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3   = snn.BatchNormTT2d(256, time_steps) if enable_bntt else nn.Identity()
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn4   = snn.BatchNormTT2d(256, time_steps) if enable_bntt else nn.Identity()
        self.lif3 = snn.Leaky(beta=beta, init_hidden=True)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn5   = snn.BatchNormTT2d(512, time_steps) if enable_bntt else nn.Identity()
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn6   = snn.BatchNormTT2d(512, time_steps) if enable_bntt else nn.Identity()
        self.lif4 = snn.Leaky(beta=beta, init_hidden=True)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn7   = snn.BatchNormTT2d(512, time_steps) if enable_bntt else nn.Identity()
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn8   = snn.BatchNormTT2d(512, time_steps) if enable_bntt else nn.Identity()
        self.lif5 = snn.Leaky(beta=beta, init_hidden=True)
        # classifier
        self.fc1  = nn.Linear(512 * 2 * 2, 4096)
        self.bn9  = snn.BatchNormTT1d(4096, time_steps) if enable_bntt else nn.ModuleList([nn.Identity() for _ in range(time_steps)])
        self.lif6 = snn.Leaky(beta=beta, init_hidden=True)
        self.fc2  = nn.Linear(4096, num_classes)

        self.pool = nn.AvgPool2d(2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2,2))
    
        def _forward_one_step(self, x, t):
            xt = x[:,t]
            B, C, H, W = xt.shape
            c1 = self.conv1(xt)
            b1 = self.bn1[t](c1)
            s1, m1 = self.lif1(b1, m1)
            s1 = self.pool(s1)
            c2 = self.conv2(s1)
            b2 = self.bn2[t](c2)
            s2, m2 = self.lif2(b2, m2)
            s2 = self.pool(s2)
            c3 = self.conv3(s2)
            b3 = self.bn3[t](c3)
            c4 = self.conv4(b3)
            b4 = self.bn4[t](c4)
            s3, m3 = self.lif3(b4, m3)
            s3 = self.pool(s3)
            c5 = self.conv5(s3)
            b5 = self.bn5[t](c5)
            c6 = self.conv6(b5)
            b6 = self.bn6[t](c6)
            s4, m4 = self.lif4(b6, m4)
            s4 = self.pool(s4)
            c7 = self.conv7(s4)
            b7 = self.bn7[t](c7)
            c8 = self.conv8(b7)
            b8 = self.bn8[t](c8)
            s5, m5 = self.lif5(b8, m5)
            s5 = self.pool(s5)
            g2 = self.adaptive_pool(s5).view(B, -1)
            f1 = self.fc1(g2)
            b9 = self.bn9[t](f1)
            s6, m6 = self.lif6(b9, m6)
            f2 = self.fc2(s6)

            return f2
