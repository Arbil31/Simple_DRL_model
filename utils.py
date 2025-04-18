import torch
import torch.nn as nn
import random
import numpy as np
import os

class AddBias(nn.Module):
    def __init__(self, bias):
        super().__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return bias
    
def init(module, weight_init, bias_init, gain = 1):
    weight_init(module.weight.data, gain = gain)
    bias_init(module.bias.data)
    return module

def init_normc(weight, gain = 1):
    weight.normal(0,1)
    weight *= gain/ torch.sqrt(weight.pow(2).sum(1, keepdim = True))

def seed_torch(seed = 2019):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHOHASHSEED'] = str(seed)

    #cpu and gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.benchmark = True

# bias = torch.randn(3)
# add_bias = AddBias(bias)
# x_fc = torch.randn(5,3)
# out_fc = add_bias(x_fc)

# print("Original bias", bias)
# print("FC output", out_fc)
