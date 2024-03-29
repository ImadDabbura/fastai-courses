import functools
import math
import torch
from .utils import listify


def annealer(func):
    functools.wraps(func)

    def annealer_wrapper(*args, **kwargs):
        return functools.partial(func, *args)

    return annealer_wrapper


@annealer
def lin_shced(start, end, pos):
    """Linear scheduler."""
    return start + pos * (end - start)


@annealer
def sched_cos(start, end, pos):
    """Cosine scheduler."""
    return start + (1 + math.cos(math.pi * (1 - pos))) * (end - start) / 2


@annealer
def sched_no(start, end, pos):
    return start


@annealer
def sched_exp(start, end, pos):
    """Exponential scheduler."""
    return start * (end / start) ** pos


def cos_1cycle_anneal(start=0.1, high=0.3, end=0.05):
    return [sched_cos(start, high), sched_cos(high, end)]


def combine_scheds(pcts, scheds):
    """Combine different scheduler of hyper-parameters during training."""
    assert sum(pcts) == 1.0
    pcts = torch.tensor([0] + listify(pcts))
    assert torch.all(pcts >= 0)
    pcts = torch.cumsum(pcts, 0)

    def _inner(pos):
        # Determine which scheduler to use
        idx = (pos >= pcts).nonzero().max()
        # Determine the actual position to be used by the chosen scheduler
        actual_pos = (pos - pcts[idx]) / (pcts[idx + 1] - pcts[idx])
        return scheds[idx](actual_pos)

    return _inner
