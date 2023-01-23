import torch
import torch.nn as nn
import torch.nn.functional as F
from model.activations import *

def activation_factory(name, inplace=True):
    if name == 'relu':
        return nn.ReLU(inplace=inplace)
    elif name == 'leakyrelu':
        return nn.LeakyReLU(0.2, inplace=inplace)
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'swish':
        return Swish()
    elif name == 'hardswish':
        return HardSwish()
    elif name == 'metaacon':
        return MetaAconC()
    elif name == 'acon':
        return AconC()
    elif name == 'linear' or name is None:
        return nn.Identity()
    else:
        raise ValueError('Not supported activation:', name)