import torch
import torch.nn as nn

import snntorch as snn

class basicSNN(nn.Module):
    '''Simple convolutional SNN, meant to operate on two channels of on and off events, with 128 x 128 resolution.'''
    def __init__(self, num_classes=11, num_channels=2, beta=0.9, threshold=1.0):
        super(basicSNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold)
        self.fc1 = nn.Linear(64 * 32 * 32, num_classes)
        self.lif3 = snn.Leaky(beta=beta, threshold=threshold)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        B, T, C, H, W = x.shape # batch size, number of time bins, number of channels(polarities), height, width
        m1 = self.lif1.reset_mem()
        m2 = self.lif2.reset_mem()
        m3 = self.lif3.reset_mem()

        out_spikes = []
        out_mem = []
        for t in range(T):
            xt = x[:, t] # b, c, h, w
            c1 = self.pool(self.conv1(xt))
            s1, m1 = self.lif1(c1, m1)
            c2 = self.pool(self.conv2(s1))
            s2, m2 = self.lif2(c2, m2)
            f  = self.fc1(s2.view(B, -1))
            s3, m3 = self.lif3(f, m3)
            out_spikes.append(s3)
            out_mem.append(m3)

        out_spikes = torch.stack(out_spikes, dim=1) # b, t, num_classes
        out_mem = torch.stack(out_mem, dim=1) # b, t, num_classes

        return out_spikes, out_mem