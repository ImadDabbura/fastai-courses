from .hooks import Hook, compute_stats


def lsuv_module(model, module, xb):
    h = Hook(module, compute_stats)

    while model(xb) is not None and abs(h.std - 1) > 1e-3:
        module.weight.data /= h.std

    while model(xb) is not None and abs(h.mean) > 1e-3:
        module.bias -= h.mean

    h.remove()
    return h.mean, h.std
