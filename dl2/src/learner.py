import torch
from .callbacks import TrainEvalCallback, ProgressCallback, Recorder
from .callbacks import (
    CancelFitException,
    CancelEpochException,
    CancelTrainException,
    CancelValidException,
    CancelBatchException,
)
from .utils import listify

"""
1. `before_fit`: called before doing anything, ideal for initial setup
`before_epoch`: called at the beginning of each epoch, useful for any behavior you need to reset at each epoch
3. `before_train`: called at the beginning of the training part of an epoch
4. `before_batch`: called at the beginning of each batch, just after drawing said batch. It can be used to do any setup necessary for the batch (like hyper-parameter scheduling) or to change the input/target before it goes in the model (change of the input with techniques like mixup for instance)
5. `after_pred`: called after computing the output of the model on the batch. It can be used to change that output before it's fed to the loss
6. `after_loss`: called after the loss has been computed, but before the backward pass. It can be used to add any penalty to the loss (AR or TAR in RNN training for instance)
7. `after_backward`: called after the backward pass, but before the update of the parameters. It can be used to do any change to the gradients before said update (gradient clipping for instance)
8. `after_step`: called after the step and before the gradients are zeroed
9. `after_cancel_batch`: reached immediately after a `CancelBatchException` before proceeding to `after_batch`
10. `after_batch`: called at the end of a batch, for any clean-up before the next one
11. `after_cancel_train`: reached immediately after a `CancelTrainException` before proceeding to `after_train`
12. `after_train`: called at the end of the training phase of an epoch
13. `before_validate`: called at the beginning of the validation phase of an epoch, useful for any setup needed specifically for validation
14. `after_cancel_validate`: reached immediately after a `CancelValidateException` before proceeding to `after_validate`
15. `after_validate`: called at the end of the validation part of an epoch
16. `after_cancel_epoch`: reached immediately after a `CancelEpochException` before proceeding to `after_epoch`
17. `after_epoch`: called at the end of an epoch, for any clean-up before the next one
18. `after_cancel_fit`: reached immediately after a `CancelFitException` before proceeding to `after_fit`
19. `after_fit`: called at the end of training, for final clean-up
"""


class Learner:
    ALL_CBS: set[str] = {
        "before_fit",
        "before_epoch",
        "before_train",
        "before_validate",
        "before_batch",
        "after_pred",
        "after_loss",
        "after_backward",
        "after_step",
        "after_cancel_batch",
        "after_batch",
        "after_cancel_train",
        "after_train",
        "after_cancel_validate",
        "after_validate",
        "after_cancel_epoch",
        "after_epoch",
        "after_cancel_fit",
        "after_fit",
    }

    def __init__(self, model, data, loss_func, opt, cbs=None, cb_funcs=None):
        self.model, self.data, self.loss_func = model, data, loss_func
        self.opt, self.training = opt, False
        # We can customize it & use progress bar or log to a file
        self.logger = print

        # Callbacks
        self.cbs = []
        self.add_cb(TrainEvalCallback)
        self.add_cb(ProgressCallback)
        self.add_cb(Recorder)
        self.add_cbs(cbs)
        self.add_cbs(cb_func() for cb_func in cb_funcs)

    def add_cbs(self, cbs):
        for cb in listify(cbs):
            self.add_cb(cb)

    def add_cb(self, cb):
        cb.set_learner(self)
        setattr(self, cb.name, cb)
        self.cbs.append(cb)

    def remove_cbs(self, cbs):
        for cb in cbs:
            self.cbs.remove(cb)

    def _one_batch(self, i, xb, yb):
        self.iter = i
        self.xb, self.yb = xb, yb
        try:
            self("before_batch")
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

    def fit(self, epochs, cbs=None, reset_opt=False):
        self.add_cbs(cbs)
        if reset_opt or not self.opt:
            self.opt = self.opt_func(self.splitter(self.model), lr=self.lr)
        self.n_epochs = epochs
        self.loss = torch.tensor(0.0)
        try:
            self("before_fit")
            for epoch in range(self.n_epochs):
                try:
                    self.epoch = epoch
                    self.dl = self.data.train_dl
                    self("before_epoch")

                    try:
                        self("before_train")
                        self._all_batches(self.data.train_dl)
                    except CancelTrainException:
                        self("after_cancel_train")
                    finally:
                        self("after_train")

                    try:
                        self.dl = self.data.valid_dl
                        self("before_validate")
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
            self.remove_cbs(cbs)

    def __call__(self, cb_name):
        assert (
            cb_name in Learner.ALL_CBS
        ), f"{cb_name} isn't a valid callback name"
        res = False
        for cb in sorted(self.cbs, key=lambda x: x._order):
            res = cb(cb_name) and res
        return res
