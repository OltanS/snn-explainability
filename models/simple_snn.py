import torch.nn as nn
import snntorch as snn

from models.base_snn import BaseSNN
from models.norms import make_norm2d_factory


class SimpleSNN(BaseSNN):
    '''Simple convolutional SNN.'''
    def __init__(self, num_classes, num_channels, time_steps, beta, norm_type):
        super(SimpleSNN, self).__init__()
        self.norm_partial = make_norm2d_factory(norm_type, time_steps=time_steps)
            
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1)
        self.norm1   = self.norm_partial(32)
        self.lif1 = snn.Leaky(beta=beta, init_hidden=True)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.norm2   = self.norm_partial(64)
        self.lif2 = snn.Leaky(beta=beta, init_hidden=True)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.norm3   = self.norm_partial(64)
        self.lif3 = snn.Leaky(beta=beta, init_hidden=True)

        self.fc1 = nn.Linear(64 * 4 * 4, num_classes)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.pool = nn.AvgPool2d(2)
        
    def _forward_one_step(self, xt, t):
        B, C, H, W = xt.shape # batch size,  number of channels(polarities), height, width
        c1 = self.conv1(xt)
        c1 = self.norm1[t](c1)
        s1 = self.lif1(c1)
        s1 = self.pool(s1)

        c2 = self.conv2(s1)
        c2 = self.norm2[t](c2)
        s2 = self.lif2(c2)
        s2 = self.pool(s2)

        c3 = self.conv3(s2)
        c3 = self.norm3[t](c3)
        s3 = self.lif3(c3)

        g2 = self.adaptive_pool(s3).view(B, -1)
        f  = self.fc1(g2)
        return f