
import torch
import torch.nn as nn
import numpy as np
import scipy

try:
    from torchdiffeq import odeint

    class OdeintWrapper(nn.Module):
        def __init__(self, model):
            super(OdeintWrapper, self).__init__()
            self.model = model
            self.nfe = 0

        def forward(self, t, x):
            self.nfe += 1
            return self.model.time_derivative(x)

        def reset(self):
            self.nfe = 0

except ImportError:
    pass
try:
    import scipy.optimize
    fsolve = scipy.optimize.fsolve

    def fsolve_gpu(func, x0, *args, **kwargs):
        shape = x0.shape
        device = x0.device
        dtype = x0.dtype
        to_gpu = lambda tensor: torch.from_numpy(tensor).view(*shape).to(device, dtype)
        to_numpy = lambda tensor: tensor.detach().cpu().numpy().reshape(-1)
        wrapped_func = lambda arg: to_numpy(func(to_gpu(arg)))
        x1 = fsolve(wrapped_func, to_numpy(x0), *args, **kwargs)
        x1 = to_gpu(x1)
        return x1

except ImportError:
    pass


discrete_autograd_mode = False
eps = None


def set_discrete_autograd_mode(flag):
    # discrete autograd or not
    global discrete_autograd_mode
    discrete_autograd_mode = flag


def set_eps(val=None):
    # threshold to use true gradient at midpoint
    global eps
    if val is not None:
        eps = val
    elif torch.get_default_dtype() == torch.float32:
        eps = 1e-6
    elif torch.get_default_dtype() == torch.float64:
        eps = 1e-12

# gradient functions


def get_grad(func):
    def grad_any(x, v):
        with torch.enable_grad():
            v = func(x.requires_grad_(True))
        return v.grad_fn(torch.ones_like(v))
    return grad_any


def grad_tanh(x, v):
    return torch.cosh(x).pow(-2)


def grad_relu(x, v):
    grad = torch.ones_like(x)
    grad[x < 0] = 0.0
    return grad


def grad_sigmoid(x, v):
    return (1 - v) * v


def grad_softplus(x, v):
    return torch.sigmoid(x)


class discrete_differential(torch.autograd.Function):
    # discrete differential for activation functions
    @staticmethod
    def forward(ctx, x1, x2, v1, v2):
        dx = x1 - x2
        dv = v1 - v2
        g = dv / dx
        ctx.save_for_backward(dx, dv, g)
        return g

    @staticmethod
    def backward(ctx, grad):
        dx, dv, g = ctx.saved_tensors
        gv1 = 1. / dx
        gv2 = -gv1
        gx2 = g / dx
        gx1 = -gx2
        return grad * gx1, grad * gx2, grad * gv1, grad * gv2


class apply_discrete_differential(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x1, x2, func, grad_fn):
        # forward path is as usual
        v1 = func(x1)
        v2 = func(x2)
        ctx.save_for_backward(x1, x2, v1, v2)
        ctx.grad_fn = grad_fn
        return v1, v2

    @staticmethod
    def backward(ctx, grad_v1, grad_v2):
        x1, x2, v1, v2 = ctx.saved_tensors
        # backward path can be discrete version or not
        if discrete_autograd_mode:
            if eps is None:
                set_eps()
            # use the true gradient when two argumants are sufficiently close.
            dp = torch.abs(x1 - x2) > eps
            grad_disc = discrete_differential().apply(x1[dp], x2[dp], v1[dp], v2[dp])
            grad_true = ctx.grad_fn((x1 + x2) / 2, (v1 + v2) / 2)
            grad_true = grad_true.clone()
            grad_true[dp] = grad_disc
            return grad_true * grad_v1, None, None, None
        else:
            # ordinary autograd for training
            grad_x1 = ctx.grad_fn(x1, v1)
            grad_x2 = ctx.grad_fn(x2, v2)
            return grad_x1 * grad_v1, grad_x2 * grad_v2, None, None


def DiscreteDifferentialLinearDecorator(cls):
    # A linear layer is as usual but accepts two arguments.
    class DecoratedLinearModule(cls):
        def _get_name(self):
            return cls.__name__ + '_DG'

        def forward(self, x1, x2=None):
            forward = super(DecoratedLinearModule, self).forward
            if x2 is None:
                return forward(x1)
            return forward(x1), forward(x2)

    return DecoratedLinearModule


def DiscreteDifferentialNonlinearDecorator(cls, func=None, grad_fn=None):
    # A non-linear layer gets a discrete gradient when two arguments are given.
    class DecoratedNonlinearModule(cls):
        def _get_name(self):
            return cls.__name__ + '_DG'

        def forward(self, x1, x2=None):
            if x2 is None:
                forward = super(DecoratedNonlinearModule, self).forward
                return forward(x1)
            return apply_discrete_differential().apply(x1, x2, func, grad_fn)

    return DecoratedNonlinearModule


def DiscreteDifferentialDecorator(cls):
    assert issubclass(cls, nn.Module)

    if cls in [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d]:
        return DiscreteDifferentialLinearDecorator(cls)

    if cls in [nn.ReLU, nn.Tanh, nn.Sigmoid, nn.Softplus]:
        if cls is nn.ReLU:
            func = torch.relu
            grad_fn = grad_relu
        elif cls is nn.Tanh:
            func = torch.tanh
            grad_fn = grad_tanh
        elif cls is nn.Sigmoid:
            func = torch.sigmoid
            grad_fn = grad_sigmoid
        elif cls is nn.Softplus:
            func = nn.functional.softplus
            grad_fn = grad_softplus
        return DiscreteDifferentialNonlinearDecorator(cls, func, grad_fn)

    raise NotImplementedError('No discrete gradient version is implemented for', cls)


class Sequential(nn.Sequential):
    # Sequential for discrete gradient
    def _get_name(self):
        return self.__class__.__name__ + '_DG'

    def forward(self, x1, x2=None):
        if x2 is None:
            for module in self:
                x1 = module(x1)
            return x1
        else:
            for module in self:
                x1, x2 = module(x1, x2)
            return x1, x2

    def grad(self, x1, x2=None):
        x1 = x1.requires_grad_(True)
        with torch.enable_grad():
            if x2 is None:
                # get a gradient when one argument is given.
                h = self(x1)
                grad = torch.autograd.grad(h.sum(), (x1,), create_graph=True)[0]
                return grad
            else:
                # get a discrete gradient when two arguments are given.
                x2 = x2.requires_grad_(True)
                h, _ = self(x1, x2)
                set_discrete_autograd_mode(True)
                grad = torch.autograd.grad(h.sum(), (x1,), create_graph=True)[0]
                set_discrete_autograd_mode(False)
                return grad


# Wrapped modules
Linear = DiscreteDifferentialDecorator(nn.Linear)
ReLU = DiscreteDifferentialDecorator(nn.ReLU)
Tanh = DiscreteDifferentialDecorator(nn.Tanh)
Sigmoid = DiscreteDifferentialDecorator(nn.Sigmoid)
Softplus = DiscreteDifferentialDecorator(nn.Softplus)
Conv1d = DiscreteDifferentialDecorator(nn.Conv1d)
# One can define a new function.
Exp = DiscreteDifferentialNonlinearDecorator(nn.Softplus, lambda x: x.exp(), lambda x, v: v)


def get_decorated_module_by_name(name):
    # get a wrapped module by name
    modules = dir(nn)
    modules_lower = [m.lower() for m in modules]
    name = name.lower()
    if name not in modules_lower:
        raise ModuleNotFoundError(name)
    idx = modules_lower.index(name)
    Module = DiscreteDifferentialDecorator(getattr(nn, modules[idx]))
    return Module


class ddmult(torch.autograd.Function):
    # discrete gradient for bilinear operation using the product rule.
    @staticmethod
    def forward(ctx, x1, y1, x2, y2):
        v1 = x1 * y1
        v2 = x2 * y2
        ctx.save_for_backward(x1, y1, x2, y2)
        return v1, v2

    @staticmethod
    def backward(ctx, grad1, grad2):
        x1, y1, x2, y2 = ctx.saved_tensors
        if discrete_autograd_mode:
            grad_x1 = 0.5 * (y1 + y2) * grad1
            grad_y1 = 0.5 * (x1 + x2) * grad1
            grad_x2 = None
            grad_y2 = None
        else:
            grad_x1 = y1 * grad1
            grad_y1 = x1 * grad1
            grad_x2 = y2 * grad2
            grad_y2 = x2 * grad2
        return grad_x1, grad_y1, grad_x2, grad_y2


class Pow2(nn.Module):
    # pow(2) makes a lower bounded function.
    def __init__(self):
        super(Pow2, self).__init__()

    def forward(self, x1, x2=None):
        if x2 is None:
            return x1.pow(2)
        return ddmult.apply(x1, x1, x2, x2)


class KineticEnergy(nn.Module):
    def __init__(self, dim):
        super(KineticEnergy, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, dim))

    def forward(self, x1, x2=None):
        if x2 is None:
            return 0.5 * (x1.pow(2) * self.weight).sum(-1, keepdim=True)
        x1sq, x2sq = ddmult.apply(x1, x1, x2, x2)
        x1sq_weighted_sum = 0.5 * (x1sq * self.weight).sum(-1, keepdim=True)
        x2sq_weighted_sum = 0.5 * (x1sq * self.weight).sum(-1, keepdim=True)
        return x1sq_weighted_sum, x2sq_weighted_sum


def periodic_pad1d(x1, x2=None, padding=1):
    # padding for periofic boundary condition.
    x1 = torch.cat([x1[..., -padding:], x1, x1[..., :padding]], dim=-1)
    if x2 is None:
        return x1
    x2 = torch.cat([x2[..., -padding:], x2, x2[..., :padding]], dim=-1)
    return x1, x2


class PeriodicPad1d(nn.Module):
    def __init__(self, padding=1):
        super(PeriodicPad1d, self).__init__()
        self.padding = padding

    def forward(self, x1, x2=None):
        return periodic_pad1d(x1, x2, padding=self.padding)


class GlobalSummation1d(nn.Module):
    # global summation to get a system energy
    def __init__(self, c=1.):
        super(GlobalSummation1d, self).__init__()
        self.c = c

    def forward(self, x1, x2=None):
        x1 = x1.sum(-1, keepdim=True) * self.c
        if x2 is None:
            return x1
        x2 = x2.sum(-1, keepdim=True) * self.c
        return x1, x2
