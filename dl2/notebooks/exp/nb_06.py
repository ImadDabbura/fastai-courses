
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: dev_nb/06_CUDA_CNN_Hooks_Init-Imad.ipynb

from exp.nb_05b import *
torch.set_num_threads(2)

def normalize_to(train, valid):
    m, s = train.mean(), train.std()
    return normalize(train, m, s), normalize(valid, m, s)

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

def flatten(x):
    return x.view(x.shape[0], -1)

def mnist_resize(x):
    return x.view(-1, 1, 28, 28)

def get_cnn_model(data):
    return nn.Sequential(
        Lambda(mnist_resize),
        nn.Conv2d(1, 8, 5, padding=2, stride=2),
        nn.ReLU(),  #14
        nn.Conv2d(8, 16, 3, padding=1, stride=2),
        nn.ReLU(),  # 7
        nn.Conv2d(16, 32, 3, padding=1, stride=2),
        nn.ReLU(),  # 4
        nn.Conv2d(32, 32, 3, padding=1, stride=2),
        nn.ReLU(),  # 2
        nn.AdaptiveAvgPool2d(1),
        Lambda(flatten),
        nn.Linear(32, data.c))

class CudaCallback(Callback):
    def begin_fit(self):
        self.model.cuda()

    def begin_batch(self):
        self.run.xb, self.run.yb = self.xb.cuda(), self.yb.cuda()

class BatchTransformXCallback(Callback):
    _order = 2

    def __init__(self, tfm):
        self.tfm = tfm

    def begin_batch(self):
        self.run.xb = self.tfm(self.xb)


def view_tfm(*size):
    def _inner(x):
        return x.view(*((-1, ) + size))

    return _inner

def get_runner(model,
               data,
               lr=0.6,
               cbs=None,
               opt_func=None,
               loss_func=F.cross_entropy):
    if opt_func is None: opt_func = optim.SGD
    opt = opt_func(model.parameters(), lr=lr)
    learn = Learner(model, opt, loss_func, data)
    return learn, Runner(cb_funcs=listify(cbs))

def children(m):
    return list(m.children())


class Hook:
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


def compute_stats(hook, module, inp, outp):
    '''Compute the means and std of each layer.'''
    if not hasattr(hook, 'stats'):
        hook.stats = ([], [])
    if not hook.is_forward:
        inp = inp[0], outp = outp[0]
    hook.stats[0].append(outp.data.mean())
    hook.stats[1].append(outp.data.std())

class ListContainer():
    def __init__(self, items):
        self.items = listify(items)

    def __getitem__(self, idx):
        if isinstance(idx, (int,slice)):
            return self.items[idx]
        if isinstance(idx[0],bool):
            assert len(idx)==len(self) # bool mask
            return [o for m,o in zip(idx,self.items) if m]
        return [self.items[i] for i in idx]

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __setitem__(self, i, o):
        self.items[i] = o

    def __delitem__(self, i):
        del(self.items[i])

    def __repr__(self):
        res = f'{self.__class__.__name__} ({len(self)} items)\n{self.items[:10]}'
        if len(self)>10: res = res[:-1]+ '...]'
        return res

class Hooks:
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

def children(m):
    return list(m.children())


class Hook:
    def __init__(self, module, func, is_forward=True, **kwargs):
        self.is_forward = is_forward
        if self.is_forward:
            self.hook = module.register_forward_hook(partial(func, self, **kwargs))
        else:
            self.hook = module.register_backward_hook(partial(func, self, **kwargs))

    def remove(self):
        self.hook.remove()

    def __del__(self):
        self.remove()

class Hooks:
    def __init__(self, modules, func, is_forward, **kwargs):
        self.hooks = [Hook(module, func, is_forward, **kwargs) for module in modules]

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

def get_cnn_layers(data, nfs, layer, **kwargs):
    nfs = [1] + nfs
    return [
        layer(nfs[i], nfs[i + 1], 5 if i == 0 else 3, **kwargs)
        for i in range(len(nfs) - 1)
    ] + [nn.AdaptiveAvgPool2d(1),
         Lambda(flatten),
         nn.Linear(nfs[-1], data.c)]


def conv_layer(ni, nf, ks=3, stride=2, **kwargs):
    return nn.Sequential(nn.Conv2d(ni, nf, ks, padding=ks // 2, stride=stride),
                         GeneralRelu(**kwargs))


class GeneralRelu(nn.Module):
    def __init__(self, leak=None, sub=None, maxv=None):
        super().__init__()
        self.leak, self.sub, self.maxv = leak, sub, maxv

    def forward(self, x):
        x = F.leaky_relu(x, self.leak) if self.leak is not None else F.relu(x)
        if self.sub is not None:
            x.sub_(self.sub)
        if self.maxv is not None:
            x.clamp_max_(self.maxv)
        return x


def init_cnn(m, uniform=False):
    f = init.kaiming_uniform_ if uniform else init.kaiming_normal_
    for l in m:
        if isinstance(l, nn.Sequential):
            f(l[0].weight, a=0.1)
            l[0].bias.data.zero_()


def get_cnn_model(data, nfs, layer, **kwargs):
    return nn.Sequential(*get_cnn_layers(data, nfs, layer, **kwargs))

def get_learn_run(nfs,
                  data,
                  lr,
                  layer,
                  cbs=None,
                  opt_func=None,
                  uniform=False,
                  **kwargs):
    model = get_cnn_model(data, nfs, layer, **kwargs)
    init_cnn(model, uniform=uniform)
    return get_runner(model, data, lr=lr, cbs=cbs, opt_func=opt_func)

from IPython.display import display, Javascript
def nb_auto_export():
    display(Javascript("""{
const ip = IPython.notebook
if (ip) {
    ip.save_notebook()
    console.log('a')
    const s = `!python ../src/notebook2script.py ${ip.notebook_name}`
    if (ip.kernel) { ip.kernel.execute(s) }
}
}"""))