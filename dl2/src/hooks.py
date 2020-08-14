from functools import partial

import torch


class Hook:
    """Register a hook into the module (forward/backward)."""

    def __init__(self, module, func, is_forward=True):
        self.is_forward = is_forward
        if self.is_forward:
            self.hook = module.register_forward_hook(partial(func, self))
        else:
            self.hook = module.register_backward_hook(partial(func, self))

    def remove(self):
        self.hook.remove()

    def __del__(self):
        self.remove()


class Hooks:
    """Register hooks on all modules."""

    def __init__(self, modules, func, is_forward):
        self.hooks = [Hook(module, func, is_forward) for module in modules]

    def __getitem__(self, idx):
        return self.hooks[idx]

    def __len__(self):
        return len(self.hooks)

    def __iter__(self):
        return iter(self.hooks)

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()

    def remove(self):
        for hook in self.hooks:
            hook.remove()

    def __del__(self):
        self.remove()


def compute_stats(hook, module, inp, outp, bins, hist_range=(0, 10)):
    """Compute the means, std, and histogram of each module."""
    if not hasattr(hook, "stats"):
        hook.stats = ([], [], [])
    if not hook.is_forward:
        inp = inp[0], outp = outp[0]
    hook.stats[0].append(outp.data.mean().cpu())
    hook.stats[1].append(outp.data.std().cpu())
    hook.stats[2].append(outp.data.cpu().histc(bins, *hist_range))


def get_hist(hook):
    """Return matrix-ready for plotting heatmap of activations."""
    return torch.stack(hook.stats[2]).t().float().log1p()


def get_min(hook, bins_range):
    """
    Compute the percentage of activations around zero from hook's histogram
    matrix.
    """
    res = torch.stack(hook.stats[2]).t().float()
    return res[slice(*bins_range)].sum(0) / res.sum(0)
