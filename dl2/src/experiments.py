from torch.utils.data import DataLoader
import torch
import torch.nn as nn


class Sequentiak(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def __call__(self, x):
        for l in self.layers:
            x = l(x)
        return x


class Optimizer:
    def __init__(self, parameters, lr):
        self.lr = lr
        self.parameters = list(parameters)

    def step(self):
        with torch.no_grad():
            for param in self.parameters:
                param -= self.lr * param.grad

    def zero_grad(self):
        for param in self.parameters:
            param.grad.data.zero_()


class DataSet:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class DataLoader:
    def __init__(self, ds, bs):
        self.ds = ds
        self.bs = bs

    def __len__(self):
        return len(self.ds)

    def __iter__(self):
        for i in range(0, len(self.ds), self.bs):
            yield self.ds[i:i+self.bs]


class Sampler:
    def __init__(self, ds, bs, shuffle):
        self.n = len(ds)
        self.bs = bs
        self.shuffle = shuffle

    def __iter__(self):
        idxs = torch.randperm(self.n) if self.shuffle else torch.arange(self.n)
        for i in range(0, self.n, bs):
            yield idxs[i:i+self.bs]


class DataLoader:
    def __init__(self, ds, bs, sampler, collate_fn):
        self.ds = ds
        self.bs = bs
        self.sampler = sampler
        self.collate = collate_fn

    def __len__(self):
        return len(self.ds)

    def __iter__(self):
        for s in self.sampler:
            yield self.collate([self.ds[i] for i in s])


def get_dls(train_ds, valid_ds, bs, **kwargs):
    return (DataLoader(train_ds, bs, shuffle=True, **kwargs),
            DataLoader(train_ds, bs*2, shuffle=False, **kwargs))
