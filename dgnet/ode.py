from .modules import *


class SepNet(nn.Module):
    def __init__(self, netV, netT):
        super(SepNet, self).__init__()
        self.netV = netV
        self.netT = netT

    def forward(self, x1, x2=None):
        if x2 is None:
            q, p = x1.chunk(2, dim=-1)
            return self.netV(q) + self.netT(p)
        q1, p1 = x1.chunk(2, dim=-1)
        q2, p2 = x2.chunk(2, dim=-1)
        v1, v2 = self.netV(q1, q2)
        t1, t2 = self.netT(p1, p2)
        return v1 + t1, v2 + t2

    def grad(self, x1, x2=None):
        return Sequential.grad(self, x1, x2)

    def gradV(self, q1, q2=None):
        return self.netV.grad(q1, q2)

    def gradT(self, p1, p2=None):
        return self.netT.grad(p1, p2)


def get_nn(input_dim, hidden_dim, nonlinearity, model=None):
    output_dim = input_dim if model == 'node' else 1
    if model == 'sephnn':
        return SepNet(
            netV=get_nn(input_dim // 2, hidden_dim, nonlinearity, model=''),
            netT=get_nn(input_dim // 2, hidden_dim, nonlinearity, model=''),
        )
    if model == 'kinhnn':
        return SepNet(
            netV=get_nn(input_dim // 2, hidden_dim, nonlinearity, model=''),
            netT=Sequential(KineticEnergy(input_dim // 2)),
        )
    Act = get_decorated_module_by_name(nonlinearity)
    model = Sequential(
        Linear(input_dim, hidden_dim),
        Act(),
        Linear(hidden_dim, hidden_dim),
        Act(),
        Linear(hidden_dim, output_dim, bias=True if model == 'node' else None),
    )
    return model


def get_symplectic_matrix(dim):
    eye = torch.eye(dim // 2)
    zero = torch.zeros((dim // 2, dim // 2))
    S = torch.cat([torch.cat([zero, eye], dim=1), torch.cat([-eye, zero], dim=1)], dim=0).to(torch.get_default_dtype())
    return S


SOLVER_LIST = [
    # available solvers in torchdiffeq
    'dopri5',
    'rk4',
    'midpoint',
    'euler',
]
SOLVER_LIST_ADDITIONAL_EXPLICIT = [
    'leapfrog',
    'leapfrog2',
]

SOLVER_LIST_ADDITIONAL_IMPLICIT = [
    'dg',
    'implicit_midpoint',
]


class DGNet(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, nonlinearity='tanh', friction=False, model='dgnet', solver='implicit'):
        super(DGNet, self).__init__()
        self.model = model
        if ',' in solver:
            self.solver, self.solver_eval = solver.split(',')
        else:
            self.solver = self.solver_eval = solver
        self.check_combination()
        self.net = get_nn(input_dim, hidden_dim, nonlinearity, model)
        self.friction = friction
        self.g = nn.Parameter(torch.zeros(input_dim // 2), requires_grad=self.friction)
        S_t = get_symplectic_matrix(input_dim).t()
        self.S_t = nn.Parameter(S_t, requires_grad=False)
        self.reset_parameters()

    def check_combination(self):
        assert self.model in ['node', 'hnn', 'sephnn', 'kinhnn']
        assert self.solver in SOLVER_LIST + SOLVER_LIST_ADDITIONAL_EXPLICIT + SOLVER_LIST_ADDITIONAL_IMPLICIT
        assert self.solver_eval in SOLVER_LIST + SOLVER_LIST_ADDITIONAL_EXPLICIT + SOLVER_LIST_ADDITIONAL_IMPLICIT

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x1, *, dt=None, x2=None, func=None):
        if func is None:
            return self.hamiltonian(x1, x2)
        if func == 'hamiltonian':
            return self.hamiltonian(x1, x2)
        if func == 'grad':
            return self.grad(x1, x2)
        if func == 'time_derivative':
            return self.time_derivative(x1, x2)
        if func == 'discrete_time_derivative':
            return self.discrete_time_derivative(x1, dt=dt, x2=x2)

    def hamiltonian(self, x1, x2=None):
        return self.net(x1, x2)

    def grad(self, x1, x2=None):
        return self.net.grad(x1, x2)

    def time_derivative(self, x1, x2=None):
        if self.model in ['node']:
            return self.net(x1)
        else:
            R_t = torch.cat([torch.zeros_like(self.g), self.g], dim=0).diag()
            return self.grad(x1, x2).mm(self.S_t - R_t)

    def discrete_time_derivative(self, x1, *, dt=None, x2=None, xtol=1.0e-12, eval_mode=False):
        # given x2, use default solver to learn, and otherwise use test solver to get state
        solver = self.solver_eval if eval_mode else self.solver
        if isinstance(dt, torch.Tensor):
            dt = dt.view(-1, *[1, ] * (len(x1.shape) - 1))
        # for implicit solvers
        if solver in SOLVER_LIST_ADDITIONAL_IMPLICIT:
            if x2 is None:
                assert not torch.is_grad_enabled()
                x2 = odeint(OdeintWrapper(self), x1, torch.tensor([0, dt]), method='midpoint')[-1]
                x2 = fsolve_gpu(lambda xp: self.discrete_time_derivative(x1, dt=None, x2=xp, eval_mode=eval_mode) - (xp - x1) / dt, x2, xtol=xtol)
                return x2
            elif solver == 'implicit_midpoint':
                dxdt = self.time_derivative((x1 + x2) / 2)
            elif solver == 'dg':
                # Get discrete gratient using Eq. (6) when x2 is given; otherwise implicitly.
                dxdt = self.time_derivative(x1, x2=x2)
            else:
                raise NotImplementedError
            return dxdt
        # for explicit solvers
        if solver in SOLVER_LIST:
            func = OdeintWrapper(self)
            x2_ = odeint(func, x1, torch.tensor([0, dt]), method=solver)[-1]
            return x2_ if x2 is None else (x2_ - x1) / dt
        elif solver == 'leapfrog':
            dt2 = dt / 2
            q, p = x1.chunk(2, dim=-1)
            p = p - dt2 * self.net.netV.grad(q)
            q = q + dt * self.net.netT.grad(p)
            p = p - dt2 * self.net.netV.grad(q)
            x2_ = torch.cat([q, p], dim=-1)
            return x2_ if x2 is None else (x2_ - x1) / dt
        elif solver == 'leapfrog2':
            dt2 = dt / 2
            q, p = x1.chunk(2, dim=-1)
            q = q + dt2 * self.net.netT.grad(p)
            p = p - dt * self.net.netV.grad(q)
            q = q + dt2 * self.net.netT.grad(p)
            x2_ = torch.cat([q, p], dim=-1)
            return x2_ if x2 is None else (x2_ - x1) / dt
        else:
            raise NotImplementedError(solver)

    def _make_sure_torch(self, arg):
        if isinstance(arg, torch.Tensor):
            return arg
        elif isinstance(arg, np.ndarray):
            return torch.tensor(arg, dtype=torch.get_default_dtype(), device=self.g.device)
        else:
            raise NotImplementedError('arg must be numpy.ndarray or torch.Tensor, but {}'.format(type(arg)))

    def get_orbit(self, x0, t_eval, tol=1.0e-12):
        target = np if isinstance(x0, np.ndarray) else torch
        x0 = self._make_sure_torch(x0)
        t_eval = self._make_sure_torch(t_eval)

        original_dim = len(x0.shape)
        if original_dim == 1:
            x0 = x0.unsqueeze(0)

        if self.solver in SOLVER_LIST:
            orbit = odeint(OdeintWrapper(self), x0, t_eval, method=self.solver)
        else:
            orbit = [x0, ]
            x1 = x0
            dts = t_eval[1:] - t_eval[:-1]
            for itr, dt in enumerate(dts):
                print(itr, '/', len(dts), end='\r')
                x2 = self.discrete_time_derivative(x1=x1, dt=dt, xtol=tol, eval_mode=True)
                orbit.append(x2)
                x1 = x2
            orbit = torch.stack(orbit, axis=0)

        if original_dim == 1:
            orbit = orbit.squeeze(1)
        if target == np:
            orbit = orbit.detach().cpu().numpy()
        return orbit
