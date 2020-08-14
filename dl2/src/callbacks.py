import re
from collections.abc import Iterable

import matplotlib.pyplot as plt
import torch
from .utils import listify


def camel2snake(name):
    """
    Convert name of callback by inserting underscores between small and capital
    letters. For example, `TestCallback` becomes `test_callback`.
    """
    pattern1 = re.compile("(.)([A-Z][a-z]+)")
    pattern2 = re.compile("([a-z0-9])([A-Z])")
    name = re.sub(pattern1, r"\1_\2", name)
    return re.sub(pattern2, r"\1_\2", name).lower()


class Callback:
    _order = 0

    def set_runner(self, run):
        self.run = run

    def __getattr__(self, k):
        return getattr(self.run, k)

    @property
    def name(self):
        """
        Returns the name of the callback after removing the word `callback` 
        and then convert it to snake (split words by underscores).
        """
        name = re.sub(r"Callback$", "", self.__class__.__name__)
        return camel2snake(name or "callback")

    def __call__(self, cb_name):
        f = getattr(self, cb_name, None)
        if f and f():
            return True
        return False


class TrainEvalCallback(Callback):
    """
    Tracks the number of iterations and epoch done and set training and eval
    modes.
    """

    def begin_fit(self):
        self.run.n_iters = 0
        self.run.pct_train = 0

    def after_batch(self):
        if self.run.training:
            self.run.n_iters += 1
            self.run.pct_train += 1 / (self.iters * self.n_epochs)

    def begin_train(self):
        self.model.train()
        self.run.training = True
        self.run.pct_train = self.epoch / self.n_epochs

    def begin_validate(self):
        self.run.training = False
        self.model.eval()


class CancelTrainException(Exception):
    """Stop training current batch and move to validation."""

    pass


class CancelEpochException(Exception):
    """Stop current epoch and move to next epoch."""

    pass


class CancelBatchException(Exception):
    """Stop current batch and move to next one."""

    pass


class CancelValidException(Exception):
    """Stop validation phase and move to next epoch"""

    pass


class CancelFitException(Exception):
    """Stop training and exit"""

    pass


class Runner:
    def __init__(self, cbs=None, cb_funcs=None):
        self.cbs = listify(cbs)
        for cb_func in listify(cb_funcs):
            cb = cb_func()
            setattr(self, cb.name, cb)
            self.cbs.append(cb)
        self.cbs = [TrainEvalCallback()] + self.cbs

    @property
    def model(self):
        return self.learn.model

    @property
    def opt(self):
        return self.learn.opt

    @property
    def loss_func(self):
        return self.learn.loss_func

    @property
    def data(self):
        return self.learn.data

    def _one_batch(self, xb, yb):
        self.xb, self.yb = xb, yb
        try:
            self("begin_batch")
            self.pred = self.model(self.xb)
            self("after_pred")
            self.loss = self.loss_func(self.pred, self.yb)
            self("after_loss")
            if not self.training:
                return
            self.loss.backward()
            self("after_backward")
            self.opt.step()
            self("after_step")
            self.opt.zero_grad()
        except CancelBatchException:
            self("after_cancel_batch")
        finally:
            self("after_batch")

    def _all_batches(self, dl):
        self.iters = len(dl)
        for xb, yb in dl:
            self._one_batch(xb, yb)

    def fit(self, epochs, learn):
        self.n_epochs = epochs
        self.learn = learn

        try:
            for cb in self.cbs:
                cb.set_runner(self)

            self("begin_fit")
            for epoch in range(self.n_epochs):
                try:
                    self("begin_epoch")
                    self.epoch = epoch

                    try:
                        self("begin_train")
                        self._all_batches(self.data.train_dl)
                    except CancelTrainException:
                        self("after_cancel_train")
                    finally:
                        self("after_train")

                    try:
                        self("begin_validate")
                        with torch.no_grad():
                            self._all_batches(self.data.valid_dl)
                    except CancelValidException:
                        self("after_cancel_validate")
                    finally:
                        self("after_validate")
                except CancelEpochException:
                    self("after_cancel_epoch")
                finally:
                    self("after_epoch")
        except CancelFitException:
            self("after_cancel_fit")
        finally:
            self("after_fit")
            self.learn = None

    def __call__(self, cb_name):
        res = False
        for cb in sorted(self.cbs, key=lambda x: x._order):
            res = cb(cb_name) or res
        return res


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

    def accumulate(self, run):
        """Evaluate metrics and accumulate them to at the epoch level."""
        bs = run.xb.shape[0]
        self.count += bs
        self.tot_loss += run.loss * bs
        for i, metric in enumerate(self.metrics):
            self.tot_metrics[i] += metric(run.pred, run.yb) * bs


class AvgStatsCallback(Callback):
    def __init__(self, metrics):
        self.train_stats = AvgStats(metrics, True)
        self.valid_stats = AvgStats(metrics, False)

    def begin_epoch(self):
        """Reset metrics/loss."""
        self.train_stats.reset()
        self.valid_stats.reset()

    def after_loss(self):
        """Evaluate metrics and accumulate them."""
        stats = self.train_stats if self.training else self.valid_stats
        with torch.no_grad():
            stats.accumulate(self.run)

    def after_epoch(self):
        print(self.train_stats)
        print(self.valid_stats)


class Recorder(Callback):
    def begin_fit(self):
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
    _order = 1

    def __init__(self, pname, sched_funcs):
        self.pname = pname
        self.sched_funcs = sched_funcs

    def begin_fit(self):
        if not isinstance(self.sched_funcs, (list, tuple)):
            self.sched_funcs = [self.sched_funcs] * len(self.opt.param_groups)

    def set_param(self):
        assert len(self.opt.param_groups) == len(self.sched_funcs)
        for pg, f in zip(self.opt.param_groups, self.sched_funcs):
            pg[self.pname] = f(self.pct_train)

    def begin_batch(self):
        if self.training:
            self.set_param()


class LR_Find(Callback):
    _order = 1

    def __init__(self, max_iter=100, min_lr=1e-6, max_lr=10):
        self.max_iter = max_iter
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.best_loss = 1e9

    def begin_batch(self):
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

    def begin_fit(self):
        self.run.model.to(self.device)

    def begin_batch(self):
        self.run.xb = self.run.xb.to(self.device)
        self.run.yb = self.run.yb.to(self.device)
