from collections import OrderedDict


def uniqueify(x, sort=False):
    res = list(OrderedDict.fromkeys(x).keys())
    if sort:
        res.sort()
    return res


class Processor:
    def process(self, items):
        return items


class CategoryProcessor(Processor):
    def __init__(self):
        self.vocab = None

    def __call__(self, items):
        # The vocab is defined on the first use, i.e. assuming that training data
        # is the first to be used.
        if self.vocab is None:
            self.vocab = uniqueify(items)
            self.otoi = {v: k for k, v in enumerate(self.vocab)}
        return [self.process(o) for o in items]

    def process(self, item):
        return self.otoi[item]

    def deprocess(self, idxs):
        assert self.vocab is not None, "Vocab is not defined"
        return [self._deprocess(idx) for idx in idxs]

    def _deprocess(self, idx):
        return self.vocab[idx]
