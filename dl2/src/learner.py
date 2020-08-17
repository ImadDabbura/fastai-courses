import torch
from .callbacks import TrainEvalCallback
from .callbacks import (
    CancelFitException, CancelEpochException, CancelTrainException,
    CancelValidException, CancelBatchException
)
from .utils import listify


class Learner:
    ALL_CBS = {
        'begin_fit', 'begin_epoch', 'begin_train', 'begin_validate',
        'begin_batch', 'after_pred', 'after_loss', 'after_backward',
        'after_step', 'after_cancel_batch', 'after_batch',
        'after_cancel_train', 'after_train', 'after_cancel_validate',
        'after_validate', 'after_cancel_epoch', 'after_epoch',
        'after_cancel_fit', 'after_fit'
    }

    def __init__(self, model, data, loss_func, opt, cbs=None, cb_funcs=None):
        self.model, self.data, self.loss_func = model, data, loss_func
        self.opt, self.in_train = opt, False
        # We can customize it and use it in something like progress bar or log to a file
        self.logger = print

        # Callbacks
        self.cbs = []
        self.add_cb(TrainEvalCallback)
        self.add_cbs(cbs)
        self.add_cbs(cb_func() for cb_func in cb_funcs)

    def add_cbs(self, cbs):
        for cb in cbs:
            self.add_cb(cb)

    def add_cb(self, cb):
        cb.set_runner(self)
        setattr(self, cb.name, cb)
        self.cbs.append(cb)

    def remove_cbs(self, cbs):
        for cb in cbs:
            self.cbs.remove(cb)

    def _one_batch(self, i, xb, yb):
        self.iter = i
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
        for i, (xb, yb) in enumerate(dl):
            self._one_batch(i, xb, yb)

    def fit(self, epochs, learn):
        self.n_epochs = epochs
        try:
            self("begin_fit")
            for epoch in range(self.n_epochs):
                try:
                    self.epoch = epoch
                    self.dl = self.data.train_dl
                    self("begin_epoch")

                    try:
                        self("begin_train")
                        self._all_batches(self.data.train_dl)
                    except CancelTrainException:
                        self("after_cancel_train")
                    finally:
                        self("after_train")

                    try:
                        self.dl = self.data.valid_dl
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
        assert cb_name in Learner.ALL_CBS
        res = False
        for cb in sorted(self.cbs, key=lambda x: x._order):
            res = cb(cb_name) and res
        return res
