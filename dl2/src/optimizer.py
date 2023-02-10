import torch
from utils import listify
from .data import compose


class SimpleOptimizer:
    def __init__(self, parameters, learning_rate):
        self.parameters = parameters
        self.learning_rate = learning_rate

    def step(self):
        with torch.no_grad():
            for param in self.parameters:
                if param.grad is not None:
                    param.data -= self.learning_rate * param.grad

    def zero_grad(self):
        for param in self.parameters:
            param.grad.data.zero_()


def maybe_update(steppers, defaults, func):
    for stepper in steppers:
        for k, v in func(defaults).items():
            if k not in defaults:
                defaults[k] = v


def get_defaults(stepper):
    return getattr(stepper, "_defaults", {})


class Optimizer:
    def __init__(self, parameters, steppers, **hyper_params):
        self.param_groups = listify(parameters)
        self.steppers = listify(steppers)
        maybe_update(self.steppers, hyper_params, get_defaults)
        if not isinstance(self.param_groups, list):
            self.param_groups = [self.param_groups]
        self.hyper_params = [{**hyper_params} for param in self.param_groups]

    def step(self):
        for param, hyper_params in self.get_params():
            compose(param, self.steppers, **hyper_params)

    def zero_grad(self):
        for param, _ in self.get_params():
            param.grad.data.zero_()

    def get_params(self):
        return [
            (param, hyper_params)
            for param_group, hyper_params in zip(
                self.param_groups, self.hyper_params
            )
            for param in param_group
            if param.grad is not None
        ]


class StatefulOptimizer(Optimizer):
    def __init__(self, parameters, steppers, stats=None, **hyper_params):
        self.stats = listify(stats)
        maybe_update(self.stats, hyper_params, get_defaults)
        super().__init__(parameters, steppers, **hyper_params)
        self.state = {}

    def step(self):
        for param, hyper_params in self.grad_params():
            if param not in self.state:
                # Create a state for p and call all the statistics to initialize it.
                self.state[param] = {}
                maybe_update(
                    self.stats,
                    self.state[param],
                    lambda o: o.init_state(param),
                )
            state = self.state[param]
            for stat in self.stats:
                state = stat.update(param, state, **hyper_params)
            compose(param, self.steppers, **state, **hyper_params)
            self.state[param] = state


class State:
    _defaults = {}

    def init_state(self, param):
        raise NotImplementedError

    def update(self, param, state, **kwargs):
        raise NotImplementedError


class AverageGrad(State):
    _defaults = {"momentum": 0.9}

    def init_state(self, param):
        return {"avg_grad": torch.zeros_like(param.grad.data)}

    def update(self, param, state, momentum, **kwargs):
        state["avg_grad"].mul_(momentum).add_(param.grad.data)
        return state


def sgd_step(param, learning_rate, **kwargs):
    param.data.add_(-learning_rate, param.grad.data)
    return param


def weight_decay(param, learning_rate, weight_decay, **kwargs):
    param.data.mul_(1 - learning_rate * weight_decay)
    return param


weight_decay._defaults = {"weight_decay": 0.0}


def l2_reg(param, learning_rate, weight_decay, **kwargs):
    param.grad.data.add_(weight_decay, param.data)
    return param


l2_reg._defaults = {"weight_decay": 0.0}


def momentum_step(param, learning_rate, avg_grad, **kwargs):
    param.data.add_(-learning_rate * avg_grad)
    return param
