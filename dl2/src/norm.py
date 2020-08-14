import math

import torch
import torch.nn as nn


class BatchNorm(nn.Module):
    """Normalize by batches, height and width for images."""

    __constants__ = ["eps"]

    def __init__(self, nf, mom, eps):
        super().__init__()
        # pytorch bn mom is opposite of what you'd expect
        # This means the beta in exponential weighted average is
        # actually 1 - mom. Therefore, beta is 0.9 here.
        self.eps = eps
        self.mom = mom
        self.gamma = nn.Parameter(torch.ones(nf, 1, 1))
        self.beta = nn.Parameter(torch.zeros(nf, 1, 1))
        self.register_buffer("means", torch.ones(1, nf, 1, 1))
        self.register_buffer("vars", torch.ones(1, nf, 1, 1))

    def update_stats(self, x):
        # Average across batches, height and width --> average across channels
        mean = x.mean((0, 2, 3), keepdim=True)
        var = x.var((0, 2, 3), keepdim=True)

        # means = (1 - mom) x means + mom x m; exponentially weighted average
        self.means.lerp_(mean, self.mom)
        self.vars.lerp_(var, self.mom)
        return mean, var

    def forward(self, x):
        if self.training:
            mean, var = self.update_stats(x)
        else:
            mean, var = self.means, self.vars
        x = (x - mean) / (var + self.eps).sqrt()
        return self.gamma * x + self.beta


class LayerNorm(nn.Module):
    """Normalize by channels, height and width for images."""

    __constants__ = ["eps"]

    def __init__(self, eps):

        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean((1, 2, 3), keepdim=True)
        var = x.var((1, 2, 3), keepdim=True)
        x = (x - mean) / (var + self.eps).sqrt()
        return self.gamma * x + self.beta


class InstanceNorm(nn.Module):
    """Normalize by height and width for images."""

    __constants__ = ["eps"]

    def __init__(self, nf, mom, eps):

        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(nf, 1, 1))
        self.beta = nn.Parameter(torch.zeros(nf, 1, 1))

    def forward(self, x):
        mean = x.mean((2, 3), keepdim=True)
        var = x.var((2, 3), keepdim=True)
        x = (x - mean) / (var + self.eps).sqrt()
        return self.gamma * x + self.beta


class RunningBatchNorm(nn.Module):
    """
    Improved version of BatchNorm that overcomes issues with small batch sizes
    with very small variances.
    """
    __constants__ = ["eps", "mom"]

    def __init__(self, nf, mom=0.1, eps=1e-5):
        super().__init__()
        self.mom, self.eps = mom, eps
        self.mults = nn.Parameter(torch.ones(nf, 1, 1))
        self.adds = nn.Parameter(torch.zeros(nf, 1, 1))
        self.register_buffer('sums', torch.zeros(1, nf, 1, 1))
        self.register_buffer('sqrs', torch.zeros(1, nf, 1, 1))
        self.register_buffer('batch', torch.tensor(0.))
        self.register_buffer('count', torch.tensor(0.))
        self.register_buffer('factor', torch.tensor(0.))
        self.register_buffer('offset', torch.tensor(0.))

    def update_stats(self, x):
        bs, nc, *_ = x.shape
        self.sums.detach_()
        self.sqrs.detach_()
        dims = (0, 2, 3)
        s = x.sum(dims, keepdim=True)
        ss = (x * x).sum(dims, keepdim=True)
        c = self.count.new_tensor(x.numel() / nc)
        mom1 = 1 - (1 - self.mom) / math.sqrt(bs-1)
        self.sums.lerp_(s, mom1)
        self.sqrs.lerp_(ss, mom1)
        self.count.lerp_(c, mom1)
        self.batch += bs
        means = self.sums / self.count
        var = (self.sqrs / self.count).sub_(means * means)

        if bool(self.batch < 20):
            var.clamp_min_(0.01)

        self.factor = self.mults / (var + self.eps).sqrt()
        self.offset = self.adds - means * self.factor

    def forward(self, x):
        if self.training:
            self.update_stats(x)
        return x * self.factor + self.offset
