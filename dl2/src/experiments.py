class ListContainer:
    def __init__(self, items):
        self.items = listify(items)

    def __getitem__(self, idx):
        if isinstance(idx, (int, slice)):
            return self.items[idx]
        elif isinstance(idx, bool):
            assert len(idx) == len(self.items)
            return [x for m, x in zip(idx, self.items) if m]
        return self.items[idx]

    def __setitem__(self, idx, x):
        self.items[idx] = x

    def __delitem__(self, x):
        self.items.remove(x)

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __repr__(self):
        res = (
            f"{self.__class__.__name__}: {len(self)} items:\n{self.items[:10]}"
        )
        if len(self) > 10:
            res = res[:-1] + "...]"
        return res


def _get_files(path, files, extension=None):
    p = Path(path)
    res = [
        p / file
        for file in files
        if not files.startswith(".")
        and ((not extension) or file.split(".")[-1].lower() in extension)
    ]
    return res


def get_files(path, recurse=False, extension=None, include=None):
    p = Path(p)
    extensions = setify(extension)
    extensions = {ext.lower() for ext in extensions}
    if recurse:
        for i, (p, d, fs) in enumerate(os.walk(p)):
            if include is not None and i == 0:
                d[:] = [o for o in d if o in include]
            else:
                d[:] = [o for o in d if not o.startswith(".")]
            res += _get_files(p, d, extensions)
        return res
    fs = [o for o in os.scandir(p) if o.is_file()]
    return _get_files(p, fs, extensions)


def compose(x, funcs, *args, order_key="_order", **kwargs):
    key = lambda o: getattr(o, "_order", 0)
    for f in sorted(listify(funcs), key=key):
        x = f(x, **kwargs)
    return x


class ItemList(ListContainer):
    def __init__(self, items, path=".", tfms=None):
        super().__init__(items)
        self.path = path
        self.tfms = tfms

    def new(self, items, cls=None):
        if cls is None:
            cls = self.__class__
        return cls(items, self.path, self.tfms)

    def get(self, idx):
        raise NotImplementedError

    def _get(self, o):
        return compose(self.get(o), self.tfms)

    def __getitem__(self, idx):
        res = super().__getitem__(idx)
        if isinstance(items, list):
            return [self._get(item) for item in res]
        return self._get(res)

    def __repr__(self):
        res = super().__repr__()
        return res + f"\nPath: {self.path}"


class ImageList(ItemList):
    @classmethod
    def fom_files(
        cls,
        path,
        recurse=True,
        extension=IMAGE_EXTENSION,
        include=None,
        **kwargs,
    ):
        return cls(
            get_files(path, recurse, extension, include), path, **kwargs
        )


class SplitData:
    def __init__(self, train, valid):
        self.train = train
        self.valid = valid

    def __repr__(self):
        return f"{self.__class__.__name__}:\nTrain: {self.train}\nValid: {self.valid}"
