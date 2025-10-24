import warnings

import torch
import torch.nn as nn
import snntorch as snn

from models.base_snn import BaseSNN
from models.norms import make_norm2d_factory, make_norm1d_factory

try:
    from torchvision.models import vgg11, VGG11_Weights
    _DEFAULT_VGG11_WEIGHTS = VGG11_Weights.IMAGENET1K_V1
except Exception:
    from torchvision.models import vgg11
    _DEFAULT_VGG11_WEIGHTS = None
    print("Warning: torchvision VGG11_Weights not available; pretrained weights cannot be loaded.")

class SpikingVGG11(BaseSNN):
    def __init__(self, num_classes, num_channels, time_steps, beta, norm_type):
        super(SpikingVGG11, self).__init__()
        
        # normalization factories
        self.norm2d_partial = make_norm2d_factory(norm_type, time_steps=time_steps)
        self.norm1d_partial = make_norm1d_factory(norm_type, time_steps=time_steps)

        self.time_steps = time_steps
        affine_flag = True
        bias_flag   = False

        # conv + bn layers
        self.conv1 = nn.Conv2d(num_channels,  64,  kernel_size=3, padding=1, bias=bias_flag)
        self.norm1   = self.norm2d_partial(64,  eps=1e-4, affine=affine_flag)
        self.lif1  = snn.Leaky(beta=beta, init_hidden=True)

        self.conv2 = nn.Conv2d(64,             128, kernel_size=3, padding=1, bias=bias_flag)
        self.norm2   = self.norm2d_partial(128, eps=1e-4, affine=affine_flag)
        self.lif2  = snn.Leaky(beta=beta, init_hidden=True)

        self.conv3 = nn.Conv2d(128,            256, kernel_size=3, padding=1, bias=bias_flag)
        self.norm3   = self.norm2d_partial(256, eps=1e-4, affine=affine_flag)
        self.lif3  = snn.Leaky(beta=beta, init_hidden=True)
        self.conv4 = nn.Conv2d(256,            256, kernel_size=3, padding=1, bias=bias_flag)
        self.norm4   = self.norm2d_partial(256, eps=1e-4, affine=affine_flag)
        self.lif4  = snn.Leaky(beta=beta, init_hidden=True)

        self.conv5 = nn.Conv2d(256,            512, kernel_size=3, padding=1, bias=bias_flag)
        self.norm5   = self.norm2d_partial(512, eps=1e-4, affine=affine_flag)
        self.lif5  = snn.Leaky(beta=beta, init_hidden=True)
        self.conv6 = nn.Conv2d(512,            512, kernel_size=3, padding=1, bias=bias_flag)
        self.norm6   = self.norm2d_partial(512, eps=1e-4, affine=affine_flag)
        self.lif6  = snn.Leaky(beta=beta, init_hidden=True)

        self.conv7 = nn.Conv2d(512,            512, kernel_size=3, padding=1, bias=bias_flag)
        self.norm7   = self.norm2d_partial(512, eps=1e-4, affine=affine_flag)
        self.lif7  = snn.Leaky(beta=beta, init_hidden=True)
        self.conv8 = nn.Conv2d(512,            512, kernel_size=3, padding=1, bias=bias_flag)
        self.norm8   = self.norm2d_partial(512, eps=1e-4, affine=affine_flag)
        self.lif8  = snn.Leaky(beta=beta, init_hidden=True)

        # classifier
        self.fc1     = nn.Linear(512, 4096, bias=bias_flag)
        self.norm9     = self.norm1d_partial(4096, eps=1e-4, affine=affine_flag)
        self.lif9    = snn.Leaky(beta=beta, init_hidden=True)
        self.fc2     = nn.Linear(4096, num_classes, bias=False)

        # pooling / dropout
        self.pool          = nn.AvgPool2d(2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc_dropout    = nn.Dropout(p=0.5)

        self.apply(self._init_weights)
        self.load_vgg11_conv_weights()

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
        b1 = self.norm1[t](c1)
        s1 = self.lif1(b1)
        s1 = self.pool(s1)

        c2 = self.conv2(s1)
        b2 = self.norm2[t](c2)
        s2 = self.lif2(b2)
        s2 = self.pool(s2)

        c3 = self.conv3(s2)
        b3 = self.norm3[t](c3)
        s3 = self.lif3(b3)
        c4 = self.conv4(s3)
        b4 = self.norm4[t](c4)
        s4 = self.lif4(b4)
        s4 = self.pool(s4)

        c5 = self.conv5(s4)
        b5 = self.norm5[t](c5)
        s5 = self.lif5(b5)
        c6 = self.conv6(s5)
        b6 = self.norm6[t](c6)
        s6 = self.lif6(b6)
        s6 = self.pool(s6)

        c7 = self.conv7(s6)
        b7 = self.norm7[t](c7)
        s7 = self.lif7(b7)
        c8 = self.conv8(s7)
        b8 = self.norm8[t](c8)
        s8 = self.lif8(b8)
        s8 = self.pool(s8)

        g2 = self.adaptive_pool(s8).view(B, -1)
        f1 = self.fc1(g2)
        b9 = self.norm9[t](f1)
        s9 = self.lif9(b9)
        s9 = self.fc_dropout(s9)
        f2 = self.fc2(s9)

        return f2
    
    @staticmethod
    def _adapt_first_conv_weight(src_w: torch.Tensor, in_channels: int) -> torch.Tensor:
        """
        Adapt a (out, 3, k, k) VGG11 first-conv weight to (out, in_channels, k, k)
        by channel-averaging and repeating, with energy scaling.
        """
        out_ch, src_in, kh, kw = src_w.shape
        assert src_in == 3, "Expected VGG11 first conv to have 3 input channels"
        if in_channels == 3:
            return src_w.clone()

        # average RGB, then repeat to target channels
        avg = src_w.mean(dim=1, keepdim=True)  # (out,1,k,k)
        w = avg.repeat(1, in_channels, 1, 1)   # (out,in_channels,k,k)

        # scale to keep similar output magnitude when in_channels != 3
        scale = 3.0 / float(in_channels)
        w *= scale
        return w

    def load_vgg11_conv_weights(self, verbose: bool = True) -> int:
        """
        Load convolutional weights from torchvision VGG11 into this SNN backbone.
        - weights="imagenet": use torchvision's VGG11_Weights.IMAGENET1K_V1 (if available).
        - weights=None and weights_path set: load state_dict from a file.
        - Returns the number of conv layers loaded.
        """
        # build a source VGG11

        if _DEFAULT_VGG11_WEIGHTS is not None:
            src = vgg11(weights=_DEFAULT_VGG11_WEIGHTS)
        else:
            src = vgg11(weights=None)
            if verbose:
                warnings.warn("No pretrained weights requested or available. Nothing to load.", UserWarning)
            return 0

        src_convs = [m for m in src.features if isinstance(m, nn.Conv2d)]
        dst_convs = [
            self.conv1, self.conv2, self.conv3, self.conv4,
            self.conv5, self.conv6, self.conv7, self.conv8
        ]
        loaded = 0
        with torch.no_grad():
            for i, (s, d) in enumerate(zip(src_convs, dst_convs)):
                sw = s.weight.data
                if i == 0:
                    dw = self._adapt_first_conv_weight(sw, d.in_channels)
                    if dw.shape != d.weight.shape:
                        raise RuntimeError(f"Adapted conv1 weight shape {dw.shape} != dest {d.weight.shape}")
                    d.weight.copy_(dw)
                    loaded += 1
                else:
                    if sw.shape == d.weight.shape:
                        d.weight.copy_(sw)
                        loaded += 1
                    else:
                        if verbose:
                            warnings.warn(f"Shape mismatch at conv{i+1}: src {sw.shape} != dst {d.weight.shape}. Skipping.", UserWarning)
        if verbose:
            print(f"[SpikingVGG11] Loaded {loaded} conv layers from VGG11.")
        return loaded