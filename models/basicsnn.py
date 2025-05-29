import torch
import torch.nn as nn

import snntorch as snn

class BasicSNN(nn.Module):
    '''Simple convolutional SNN, meant to operate on two channels of on and off events, with 128 x 128 resolution.'''
    def __init__(self, num_classes=11, num_channels=2, beta=0.9, time_steps=30):
        super(BasicSNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1)
        self.lif1 = snn.Leaky(beta=beta)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.lif2 = snn.Leaky(beta=beta)
        self.fc1 = nn.Linear(64 * 4 * 4, num_classes)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        B, T, C, H, W = x.shape # batch size, number of time bins, number of channels(polarities), height, width
        m1 = self.lif1.reset_mem()
        m2 = self.lif2.reset_mem()

        outputs = []
        for t in range(T):
            xt = x[:, t] # b, c, h, w
            c1 = self.pool(self.conv1(xt))
            s1, m1 = self.lif1(c1, m1)
            c2 = self.pool(self.conv2(s1))
            s2, m2 = self.lif2(c2, m2)

            g2 = self.adaptive_pool(s2).view(B, -1)  # Global average pooling
            f  = self.fc1(g2)
            outputs.append(f)

        return torch.stack(outputs, dim=1)  # B, T, num_classes