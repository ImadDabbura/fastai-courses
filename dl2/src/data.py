import mimetypes
import os
from pathlib import Path

import numpy as np
import PIL
import torch
from torch.utils.data import DataLoader

from .utils import ListContainer, listify, setify

IMAGE_EXTENSIONS = [
    k for k, v in mimetypes.map_types.items() if v.startswith("image/")
]


def get_dls(train_ds, valid_ds, bs, **kwargs):
    """
    Returns two data loaders: 1 for training and 1 for 1 for validation.
    """
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True, **kwargs),
        DataLoader(valid_ds, batch_size=bs * 2, shuffle=False, **kwargs),
    )


def _get_files(p, fs, extensions=None):
    """Get filenames in `path` that have extension `extensions`."""
    p = Path(p)
    res = [
        p / f
        for f in fs
        if not f.startswith(".")
        and ((not extensions) or f'.{f.split(".")[-1].lower()}' in extensions)
    ]
    return res


def get_files(path, extensions=None, include=None, recurse=False):
    """
    Get filenames in `path` that have extension `extensions` starting
    with `path` and optionally recurse to subdirectories.
    """
    path = Path(path)
    extensions = setify(extensions)
    extensions = {e.lower() for e in extensions}
    if recurse:
        res = []
        for i, (p, d, fs) in enumerate(os.walk(path)):
            if include is not None and i == 0:
                d[:] = [o for o in d if o in include]
            else:
                d[:] = [o for o in d if not o.startswith(".")]
            res += _get_files(p, fs, extensions)
        return res
    else:
        fs = [o.name for o in os.scandir(path) if o.is_file()]
        return _get_files(path, fs, extensions)


ta

def compose(x, funcs, *args, order="_order", **kwargs):
    """
    Applies functions/transformations in `funcs` to the input `x` in the order
    of `order`.
    """
    key = lambda x: getattr(x, order, 0)
    for func in sorted(listify(funcs), key=key):
        x = func(x, *args, **kwargs)
    return x


class ItemList(ListContainer):
    def __init__(self, items, path=".", tfms=None):
        super().__init__(items)
        self.path = path
        self.tfms = tfms

    def __repr__(self):
        return super().__repr__() + f"\nPath: {self.path}"

    def new(self, items, cls=None):
        if cls is None:
            cls = self.__class__
        return cls(items, self.path, self.tfms)

    def get(self, item):
        return item

    def _get(self, item):
        return compose(self.get(item), self.tfms)

    def __getitem__(self, idx):
        items = super().__getitem__(idx)
        if isinstance(idx, list):
            return [self._get(item) for item in items]
        return self._get(items)


class ImageList(ItemList):
    @classmethod
    def from_files(
        cls,
        path,
        extensions=IMAGE_EXTENSIONS,
        include=None,
        recurse=True,
    ):
        return cls(
            get_files(path, extensions, include, recurse), path, **kwargs
        )

    def get(self, fn):
        return PIL.Image.open(fn)


# Having transforms as classes allows to have `_order` in each one that
# will be used to sort them when applying transforms in `compose` function
class Transform:
    _order = 0


class MakeRGB(Transform):
    def __call__(self, item):
        # if the image is already rgb, then PIL does not do anything
        return item.convert("RGB")


def random_splitter(f_name, p_valid):
    return np.random.random() < p_valid


def grandparent_splitter(fn, valid_name="valid", train_name="train"):
    """
    Split items based on whether they fall under validation or training
    direcotories. This assumes that the directory structure is
    train/label/items or valid/label/items.
    """
    gp = fn.parent.parent.name
    if gp == valid_name:
        return True
    elif gp == train_name:
        return False
    return


def split_by_func(items, func):
    mask = [func(o) for o in items]
    # `None` values will be filtered out
    val = [o for o, m in zip(items, mask) if m]
    train = [o for o, m in zip(items, mask) if m is False]
    return train, val


class SplitData:
    def __init__(self, train, valid):
        self.train = train
        self.valid = valid

    def __getattr__(self, k):
        return getattr(self.train, k)

    # This is needed if we want to pickle SplitData and be able to load it back without recursion errors
    def __setstate__(self, data):
        self.__dict__.update(data)

    @classmethod
    def split_by_func(cls, il, func):
        lists = map(il.new, split_by_func(il.items, func))
        return cls(*lists)

    def to_databunch(self, bs, c_in, c_out, **kwargs):
        dls = get_dls(self.train, self.valid, bs, **kwargs)
        return DataBunch(*dls, c_in=c_in, c_out=c_out)

    def __repr__(self):
        return f"{self.__class__.__name__}\nTrain: {self.train}\nValid: {self.valid}\n"


def parent_labeler(fn):
    return fn.parent.name


def _label_by_func(ds, func, cls=ItemList):
    return cls([func(o) for o in ds.items], path=ds.path)


class LabeledData:
    def __init__(self, x, y, proc_x=None, proc_y=None):
        self.x = self.process(x, proc_x)
        self.y = self.process(y, proc_y)
        self.proc_x = proc_x
        self.proc_y = proc_y

    def process(self, il, proc):
        return il.new(compose(il.items, proc))

    def __repr__(self):
        return f"{self.__class__.__name__}\nx: {self.x}\ny: {self.y}\n"

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)

    def x_obj(self, idx):
        return self.obj(self.x, idx, self.proc_x)

    def y_obj(self, idx):
        return self.obj(self.y, idx, self.proc_y)

    def obj(self, items, idx, procs):
        isint = isinstance(idx, int) or (
            isinstance(idx, torch.LongTensor) and not idx.ndim
        )
        item = items[idx]
        for proc in reversed(listify(procs)):
            item = proc._deprocess(item) if isint else proc.deprocess(item)
        return item

    @classmethod
    def label_by_func(cls, il, func, proc_x=None, proc_y=None):
        return cls(il, _label_by_func(il, func), proc_x=proc_x, proc_y=proc_y)


def label_by_func(sd, func, proc_x=None, proc_y=None):
    train = LabeledData.label_by_func(
        sd.train, func, proc_x=proc_x, proc_y=proc_y
    )
    valid = LabeledData.label_by_func(
        sd.valid, func, proc_x=proc_x, proc_y=proc_y
    )
    return SplitData(train, valid)


class DataBunch:
    def __init__(self, train_dl, valid_dl, c_in=None, c_out=None):
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.c_in = c_in
        self.c_out = c_out

    @property
    def train_ds(self):
        return self.train_dl.dataset

    @property
    def valid_ds(self):
        return self.valid_dl.dataset


class TextList(ItemList):
    @classmethod
    def from_files(
        cls, path, extensions=".txt", recurse=True, include=None, **kwargs
    ):
        return cls(
            get_files(path, extensions, recurse=recurse, include=include),
            path,
            **kwargs,
        )

    def get(self, i):
        if isinstance(i, Path):
            return read_file(i)
        return i
