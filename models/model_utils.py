import torch.nn as nn

class BatchNorm2dTTWrapper(nn.Module):
    def __init__(self, input_features, time_steps=None, eps=1e-5, affine=False):
        super(BatchNorm2dTTWrapper, self).__init__()
        # register this as a real submodule
        self.bn = nn.BatchNorm2d(input_features, eps=eps, affine=affine)
    def __getitem__(self, idx):
        return self.bn

class BatchNorm1dTTWrapper(nn.Module):
    def __init__(self, input_features, time_steps=None, eps=1e-5, affine=False):
        super(BatchNorm1dTTWrapper, self).__init__()
        self.bn = nn.BatchNorm1d(input_features, eps=eps, affine=affine)
    def __getitem__(self, idx):
        return self.bn