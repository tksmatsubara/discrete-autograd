# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import numpy as np
import torch
import scipy.optimize
fsolve = scipy.optimize.fsolve
import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from utils import to_pickle, from_pickle


class PDENd:
    def __init__(self, ndiv=50, width=1., device='cpu'):
        self.ndiv = ndiv
        self.width = width
        self.dx = width / self.ndiv
        self.device = device
        self.dim = None

    def _pow(self, u, nu, arg):
        if nu is None:
            return u**arg
        if arg == 1:
            return (u + nu) * 0.5
        if arg == 2:
            return (u * u + u * nu + nu * nu) / 3.
        if arg == 3:
            return (nu * nu * nu + nu * nu * u + nu * u * u + u * u * u) * 0.25
        raise NotImplementedError

    def dudtdiff(self, u, nu, dt, t):
        dudt_right = self.dudt(u=u, nu=nu, t=t)
        dudt_left = (nu - u) / dt
        diff = dudt_left - dudt_right
        return diff

    def dvdmint(self, x0, t_eval):
        if len(x0.shape) == 1:
            x0 = x0.reshape(1, *x0.shape)
        if len(x0.shape) == 2:
            x0 = x0.reshape(1, *x0.shape)
        res = [x0, ]
        dts = t_eval[1:] - t_eval[:-1]
        xn = x0
        for i in range(t_eval.size - 1):
            print(i, '/', t_eval.size - 1, end='\r')
            xn1 = xn + dts[i] * self.dudt(u=xn, t=t_eval[i])
            xn1 = fsolve(lambda xnp: self.dudtdiff(u=xn, nu=xnp.reshape(*xn.shape), dt=dts[i], t=t_eval[i]).reshape(-1), xn1.reshape(-1), xtol=1.0e-10)
            xn = xn1.reshape(*xn.shape)
            res.append(xn)
        res = np.stack(res, axis=0)
        return res

    def _global_integral(self, u):
        return u.view(*u.shape[:-self.dim], -1).sum(-1) * self.dx**self.dim


class PDE1dPeriodic(PDENd):
    def __init__(self):
        self.dim = 1
        self.kernel_dx2 = torch.tensor(
            [1., -2., 1.],
            dtype=torch.get_default_dtype(),
            device=self.device).view(1, 1, 3) / (self.dx * self.dx)
        self.kernel_dx_central = torch.tensor(
            [-1., 0., 1.],
            dtype=torch.get_default_dtype(),
            device=self.device).view(1, 1, 3) / (2. * self.dx)
        self.kernel_dx_forward = torch.tensor(
            [0., -1., 1.],
            dtype=torch.get_default_dtype(),
            device=self.device).view(1, 1, 3) / self.dx
        self.kernel_dx_backward = torch.tensor(
            [-1., 1., 0.],
            dtype=torch.get_default_dtype(),
            device=self.device).view(1, 1, 3) / self.dx

    def _D(self, u):
        shape = u.shape
        u = u.view(-1, 1, shape[-1])
        u_pad = torch.cat([u[..., -1:], u, u[..., :1]], dim=-1)
        u_conv = torch.nn.functional.conv1d(u_pad, self.kernel_dx_central, padding=0)
        u_conv = u_conv.view(shape)
        return u_conv

    def _D2(self, u):
        shape = u.shape
        u = u.view(-1, 1, shape[-1])
        u_pad = torch.cat([u[..., -1:], u, u[..., :1]], dim=-1)
        u_conv = torch.nn.functional.conv1d(u_pad, self.kernel_dx2, padding=0)
        u_conv = u_conv.view(shape)
        return u_conv

    def _Dx2(self, u):
        shape = u.shape
        u = u.view(-1, 1, shape[-1])
        u_pad = torch.cat([u[..., -1:], u, u[..., :1]], dim=-1)
        conved_u_forward = torch.nn.functional.conv1d(u_pad, self.kernel_dx_forward, padding=0)
        conved_u_backward = torch.nn.functional.conv1d(u_pad, self.kernel_dx_backward, padding=0)
        u2 = (conved_u_forward**2 + conved_u_backward**2) / 2
        u2 = u2.view(shape)
        return u2


class CahnHilliardNd(PDENd):
    def __init__(self, ndiv=50, width=1., a=1., b=1., gamma=0.005, device='cpu'):
        PDENd.__init__(self, ndiv=ndiv, width=width, device=device)
        self.a = a
        self.b = b
        self.gamma = gamma

    def dudt(self, u, *, nu=None, t=None):
        u = torch.from_numpy(u).to(dtype=torch.get_default_dtype(), device=self.device)
        if nu is not None:
            nu = torch.from_numpy(nu).to(dtype=torch.get_default_dtype(), device=self.device)
        u1 = self._pow(u, nu, 1)
        u3 = self._pow(u, nu, 3)
        dudt = self._D2(- self.a * u1 + self.b * u3 - self.gamma * self._D2(u1))
        return dudt.detach().cpu().numpy()

    def get_energy(self, u):
        u = torch.tensor(u, dtype=torch.get_default_dtype(), device=self.device)
        local_energy = -0.5 * self.a * u**2 + 0.25 * self.b * u**4 + 0.5 * self.gamma * self._Dx2(u)
        energy = self._global_integral(local_energy)
        return energy.detach().cpu().numpy()


class KdVNd(PDENd):
    def __init__(self, ndiv=50, width=1., a=6., b=1., device='cpu'):
        PDENd.__init__(self, ndiv=ndiv, width=width, device=device)
        self.a = a
        self.b = b

    def dudt(self, u, *, nu=None, t=None):
        u = torch.from_numpy(u).to(dtype=torch.get_default_dtype(), device=self.device)
        if nu is not None:
            nu = torch.from_numpy(nu).to(dtype=torch.get_default_dtype(), device=self.device)
        u1 = self._pow(u, nu, 1)
        u2 = self._pow(u, nu, 2)
        dudt = self._D(- self.a / 2 * u2 + self.b * self._D2(u1))
        return dudt.detach().cpu().numpy()

    def get_energy(self, u):
        u = torch.tensor(u, dtype=torch.get_default_dtype(), device=self.device)
        local_energy = -self.a / 6. * u**3 - self.b / 2. * self._Dx2(u)
        energy = self._global_integral(local_energy)
        return energy.detach().cpu().numpy()


class CahnHilliard1d(CahnHilliardNd, PDE1dPeriodic):
    def __init__(self, ndiv=50, width=1., a=1., b=1., gamma=0.005, device='cpu'):
        CahnHilliardNd.__init__(self, ndiv=ndiv, width=width, a=a, b=b, gamma=gamma, device=device)
        PDE1dPeriodic.__init__(self)


class KdV1d(KdVNd, PDE1dPeriodic):
    def __init__(self, ndiv=50, width=1., a=-6., b=1., device='cpu'):
        KdVNd.__init__(self, ndiv=ndiv, width=width, a=a, b=b, device=device)
        PDE1dPeriodic.__init__(self)


def make_ch_dataset(name='ch', test_split=0.1, device='cpu', verbose=False, long_data=False):
    torch.set_default_dtype(torch.float64)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    rng = np.random.get_state()
    np.random.seed(0)

    ndata = 100
    M = 50
    width = 1.
    gamma = 0.0005
    dt = 0.0001
    N = 500
    if long_data:
        np.random.seed(100)
        ndata = 5
        test_split = 1.0
        N = 5000
    t_eval = np.arange(0, N + 1) * dt

    ch1d = CahnHilliard1d(ndiv=M, width=width, device=device, gamma=gamma)
    u_results = []
    for i in range(ndata):
        if verbose:
            print('generating CH dataset,', i, '/', ndata, end='\r')
        x0 = (np.random.rand(1, 1, M) - 0.5) * 0.1
        u_result = ch1d.dvdmint(x0, t_eval)
        u_result = u_result.reshape(-1, 1, M)
        u_results.append(u_result)
    u_results = np.stack(u_results, axis=0)

    data = {}
    dudt = ch1d.dudt(u_results.reshape(-1, 1, M)).reshape(u_results.shape)
    energy = ch1d.get_energy(u_results.reshape(-1, 1, M)).reshape(u_results.shape[:2])
    ntrain = int(ndata * (1 - test_split))
    data['u'] = u_results[:ntrain]
    data['test_u'] = u_results[ntrain:]
    data['dudt'] = dudt[:ntrain]
    data['test_dudt'] = dudt[ntrain:]
    data['energy'] = energy[:ntrain]
    data['test_energy'] = energy[ntrain:]
    data['dt'] = dt
    data['t_eval'] = t_eval
    data['dx'] = ch1d.dx
    data['M'] = M
    data['model'] = ch1d

    np.random.set_state(rng)
    return data


def make_kdv_dataset(name='kdv', test_split=0.1, device='cpu', verbose=False, long_data=False):
    torch.set_default_dtype(torch.float64)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    rng = np.random.get_state()
    np.random.seed(1)

    ndata = 100
    M = 50
    width = 10.
    dt = 0.001
    N = 500
    a = -6.
    b = 1.
    if long_data:
        np.random.seed(100)
        ndata = 5
        test_split = 1.0
        N = 5000
    t_eval = np.arange(0, N + 1) * dt
    x = width * np.arange(M) / M

    np.sech = lambda a: 1 / np.cosh(a)
    kdv1d = KdV1d(width=width, ndiv=M, a=a, b=b, device='cpu')
    u_results = []
    for i in range(ndata):
        if verbose:
            print('generating KdV dataset,', i, '/', ndata, end='\r')
        k1, k2 = np.random.uniform(0.5, 2.0, 2)
        d1 = np.random.uniform(0.2, 0.3, 1)
        d2 = d1 + np.random.uniform(0.2, 0.5, 1)
        x = width * np.arange(M) / M
        u0 = 0
        u0 += (-6. / a) * 2 * k1**2 * np.sech(k1 * (x - width * d1))**2
        u0 += (-6. / a) * 2 * k2**2 * np.sech(k2 * (x - width * d2))**2
        shift = np.random.randint(0, M)
        u0 = np.concatenate([u0[M:], u0[:M]], axis=-1)
        if np.random.randint(0, 2) == 1:
            u0 = u0[::-1].copy()
        u_result = kdv1d.dvdmint(u0, t_eval)
        u_result = u_result.reshape(-1, 1, M)
        u_results.append(u_result)
    u_results = np.stack(u_results, axis=0)

    data = {}
    dudt = kdv1d.dudt(u_results.reshape(-1, 1, M)).reshape(u_results.shape)
    energy = kdv1d.get_energy(u_results.reshape(-1, 1, M)).reshape(u_results.shape[: 2])
    ntrain = int(ndata * (1 - test_split))
    data['u'] = u_results[: ntrain]
    data['test_u'] = u_results[ntrain:]
    data['dudt'] = dudt[: ntrain]
    data['test_dudt'] = dudt[ntrain:]
    data['energy'] = energy[: ntrain]
    data['test_energy'] = energy[ntrain:]
    data['dt'] = dt
    data['t_eval'] = t_eval
    data['dx'] = kdv1d.dx
    data['M'] = M
    data['model'] = kdv1d

    np.random.set_state(rng)
    return data


def make_dataset(name='ch', test_split=0.1, device='cpu', verbose=False):
    print(name)
    if name.startswith('ch'):
        return make_ch_dataset(name=name, test_split=test_split, device=device, verbose=verbose, long_data='_long' in name)
    if name.startswith('kdv'):
        return make_kdv_dataset(name=name, test_split=test_split, device=device, verbose=verbose, long_data='_long' in name)
    raise NotImplementedError(name)


def get_dataset(experiment_name, save_dir, **kwargs):
    '''Returns a PDE dataset.'''
    path = '{}/{}-dataset.pkl'.format(save_dir, experiment_name)

    try:
        data = from_pickle(path)
        print("Successfully loaded data from {}".format(path))
    except:
        print("Had a problem loading data from {}. Rebuilding dataset...".format(path))
        data = make_dataset(experiment_name, **kwargs)
        to_pickle(data, path)
        os.makedirs('{}/data/'.format(save_dir), exist_ok=True)
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        u_all = np.concatenate([data['u'], data['test_u']], axis=0)
        energy_all = np.concatenate([data['energy'], data['test_energy']], axis=0)
        mass_all = u_all.sum(-1).squeeze(-1)
        for idx in range(len(u_all)):
            u = u_all[idx]
            energy = energy_all[idx]
            mass = mass_all[idx]
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(6., 6.), facecolor='white')
            t = data['t_eval']
            M = u.shape[-1]
            y = np.arange(M) / M
            T, Y = np.meshgrid(t, y)
            if experiment_name.startswith('ch'):
                ax1.pcolormesh(T, Y, u.squeeze(1).T, cmap='seismic', vmin=-1, vmax=1)
            else:
                ax1.pcolormesh(T, Y, u.squeeze(1).T, cmap='seismic')
            ax1.set_aspect('auto')
            ax1.set_yticks((0 - .5 / M, 1 - .5 / M))
            ax1.set_yticklabels((0, 1))
            ax2.plot(t, energy)
            ax3.plot(t, mass)
            ax3.set_xticks((t[0], t[-1]))
            ax3.set_xticklabels((t[0], t[-1]))
            fig.savefig('{}/data/data_{}_{:02d}.png'.format(save_dir, experiment_name, idx))
            plt.close()
    return data
