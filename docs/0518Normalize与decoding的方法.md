# 0518Normalize与decoding的方法

import torch

import torch.nn as nn

class LayerNorm(nn.Module):

def __init__(self, dim, eps=1e-6):

super(LayerNorm, self).__init__()

self.eps = eps

self.weight = nn.Parameter(torch.ones(dim))

self.bias = nn.Parameter(torch.zeros(dim))

def forward(self, x):

mean = x.mean(-1, keepdim=True)

std = x.std(-1, keepdim=True, unbiased=False)

return self.weight * (x - mean) / (std + self.eps) + self.bias

class RMSNorm(torch.nn.Module):

def __init__(self, dim: int, eps: float = 1e-6):

super().__init__()

self.eps = eps

self.weight = nn.Parameter(torch.ones(dim))

def _norm(self, x):

return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

def forward(self, x):

output = self._norm(x.float()).type_as(x)

return output * self.weight