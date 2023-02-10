import re
import time
from collections.abc import Iterable
from functools import partial

import matplotlib.pyplot as plt
import torch
from fastprogress.fastprogress import format_time, master_bar, progress_bar

from .utils import listify


class CancelFitException(Exception):
    """Stop training and exit"""


class CancelEpochException(Exception):
    """Stop current epoch and move to next epoch."""


class CancelTrainException(Exception):
    """Stop training current batch and move to validation."""


class CancelValidException(Exception):
    """Stop validation phase and move to next epoch"""


class CancelBatchException(Exception):
    """Stop current batch and move to next batch."""


class Callback:
    _order = 0

    def set_learner(self, learner):
        self.learner = learner

    def __getattr__(self, k):
        return getattr(self.learner, a bk)

    @property
    def name(self):
        """
        Returns the name of the callback after removing the word `callback`
        and then convert it to snake (split words by underscores).
        """
        name = re.sub(r"Callback$", "", self.__class__.__name__)
        return Callback.camel2snake(name or "callback")

    def __call__(self, cb_name):
        f = getattr(self, cb_name, None)
        if f and f():
            return True
        return False

    @staticmethod
    def camel2snake(name):
        """
        Convert name of callback by inserting underscores between small and capital
        letters. For example, `TestCallback` becomes `test_callback`.
        """
        pattern1 = re.compile("(.)([A-Z][a-z]+)")
        pattern2 = re.compile("([a-z0-9])([A-Z])")
        name = re.sub(pattern1, r"\1_\2", name)
        return re.sub(pattern2, r"\1_\2", name).lower()


class TrainEvalCallback(Callback):
    """
    Tracks the number of iterations and epoch done and set training and eval
    modes.
    """

    _order = -10

    def before_fit(self):
        self.learner.n_iters = 0
        self.learner.pct_train = 0

    def after_batch(self):
        if self.learner.training:
            self.learner.n_iters += 1
            self.learner.pct_train += 1 / (self.iters * self.n_epochs)

    def before_train(self):
        self.model.train()
        self.learner.training = True
        self.learner.pct_train = self.epoch / self.n_epochs

    def before_validate(self):
        self.learner.training = False
        self.model.eval()


class AvgStats:
    def __init__(self, metrics, training=True):
        self.metrics = listify(metrics)
        self.training = training

    def reset(self):
        self.tot_loss = 0
        self.count = 0
        self.tot_metrics = [0.0] * len(self.metrics)

    @property
    def all_stats(self):
        """Returns a list of both loss and metrics."""
        return [self.tot_loss.item()] + self.tot_metrics

    @property
    def avg_stats(self):
        """Returns the average of loss/metrics."""
        return [o / self.count for o in self.all_stats]

    def __repr__(self):
        if not self.count:
            return ""
        return f"{'train' if self.training else 'valid'}: {self.avg_stats}"

    def accumulate(self, learner):
        """Evaluate metrics and accumulate them to at the epoch level."""
        bs = learner.xb.shape[0]
        self.count += bs
        self.tot_loss += learner.loss * bs
        for i, metric in enumerate(self.metrics):
            self.tot_metrics[i] += metric(learner.pred, learner.yb) * bs


class AvgStatsCallback(Callback):
    _order = -10

    def __init__(self, metrics):
        self.train_stats = AvgStats(metrics, True)
        self.valid_stats = AvgStats(metrics, False)

    def before_fit(self):
        metrics_names = ["loss"] + [
            metric.__name__ for metric in self.train_stats.metrics
        ]
        names = (
            ["epoch"]
            + [f"train_{name}" for name in metrics_names]
            + [f"valid_{name}" for name in metrics_names]
            + ["time"]
        )
        self.logger(names)

    def before_epoch(self):
        """Reset metrics/loss."""
        self.train_stats.reset()
        self.valid_stats.reset()
        self.start_time = time.time()

    def after_loss(self):
        """Evaluate metrics and accumulate them."""
        stats = self.train_stats if self.training else self.valid_stats
        with torch.no_grad():
            stats.accumulate(self.learner)

    def after_epoch(self):
        stats = [str(self.epoch)]
        for o in [self.train_stats, self.valid_stats]:
            stats += [f"{v:.6f}" for v in o.avg_stats]
        stats += [format_time(time.time() - self.start_time)]
        self.logger(stats)


class ProgressCallback(Callback):
    """Add progress bar as logger for tracking metrics."""

    _order = -1

    def before_fit(self):
        self.mbar = master_bar(range(self.learner.n_epochs))
        self.mbar.on_iter_begin()
        self.learner.logger = partial(self.mbar.write, table=True)

    def after_fit(self):
        self.mbar.on_iter_end()

    def after_batch(self):
        self.pb.update(self.learner.iter)

    def before_epoch(self):
        self.set_pb()

    def before_validate(self):
        self.set_pb()

    def set_pb(self):
        self.pb = progress_bar(self.learner.dl, parent=self.mbar)
        self.mbar.update(self.epoch)


class Recorder(Callback):
    _order = 50

    def before_fit(self):
        self.lrs = [[] for _ in self.opt.param_groups]
        self.losses = []

    def after_batch(self):
        if not self.training:
            return
        for pg, lr in zip(self.opt.param_groups, self.lrs):
            lr.append(pg["lr"])
        self.losses.append(self.loss.detach().cpu())

    def plot_lr(self, pgid=-1):
        plt.plot(self.lrs[pgid])

    def plot_loss(self, skip_last=0):
        n = len(self.losses) - skip_last
        plt.plot(self.losses[:n])

    def plot(self, skip_last=0, pgid=-1):
        losses = [o.item() for o in self.losses]
        lrs = self.lrs[pgid]
        n = len(losses) - skip_last
        plt.xscale("log")
        plt.plot(lrs[:n], losses[:n])


class ParamScheduler(Callback):
    _order = 60

    def __init__(self, pname, sched_funcs):
        self.pname = pname
        self.sched_funcs = sched_funcs

    def before_fit(self):
        if not isinstance(self.sched_funcs, (list, tuple)):
            self.sched_funcs = [self.sched_funcs] * len(self.opt.param_groups)

    def set_param(self):
        assert len(self.opt.param_groups) == len(self.sched_funcs)
        for pg, f in zip(self.opt.param_groups, self.sched_funcs):
            pg[self.pname] = f(self.pct_train)

    def before_batch(self):
        if self.training:
            self.set_param()


class LR_Find(Callback):
    _order = 60

    def __init__(self, max_iter=100, min_lr=1e-6, max_lr=10):
        self.max_iter = max_iter
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.best_loss = 1e9

    def before_batch(self):
        if not self.training:
            return
        pos = self.n_iters / self.max_iter
        lr = self.min_lr * (self.max_lr / self.min_lr) ** pos
        for pg in self.opt.param_groups:
            pg["lr"] = lr

    def after_step(self):
        if self.n_iters >= self.max_iter or self.loss > self.best_loss * 10:
            raise CancelFitException()
        elif self.loss < self.best_loss:
            self.best_loss = self.loss


class CudaCallback(Callback):
    def __init__(self, device):
        self.device = device

    def before_fit(self):
        self.learner.model.to(self.device)

    def before_batch(self):
        self.learner.xb = self.learner.xb.to(self.device)
        self.learner.yb = self.learner.yb.to(self.device)


class Mixup(Callback):
    _order = 90

    def __init__(self, alpha=0.4):
        self.distrib = torch.distributions.beta.Beta(
            torch.tensor([alpha], torch.tensor([alpha]))
        )

    def before_fit(self):
        self.learner.loss_func, self.old_loss = (
            self.loss_func,
            self.learner.loss_func,
        )

    def after_fit(self):
        self.learner.loss_func = self.old_loss

    def before_batch(self):
        λ = self.distrib.sample((self.learner.xb.size(0),)).to(
            self.learner.xb.device
        )
        λ = torch.stack([λ, 1 - λ], dim=1)
        self.λ = λ.max(1)[0].view(-1, 1, 1, 1)
        shuffle = torch.randperm(self.learner.xb.size(0))
        self.learner.xb = self.learner.xb * self.λ + self.learner.xb[
            shuffle
        ] * (1 - self.λ)
        self.yb1 = self.learner.yb[shuffle]

    def after_batch(self):
        loss = loss(pred, self.yb)
        loss2 = loss(pred, self.yb1)
        loss = loss * self.λ + loss2 * (1 - self.λ)
