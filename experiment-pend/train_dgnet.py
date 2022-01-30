# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_default_dtype(torch.float32)
import argparse
import numpy as np

import os
import sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from hnn import HNN
from data import get_dataset
from utils import L2_loss, to_pickle
import dgnet


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--noretry', dest='noretry', action='store_true', help='not do a finished trial.')
    # network, experiments
    parser.add_argument('--input_dim', default=2, type=int, help='dimensionality of input tensor')
    parser.add_argument('--hidden_dim', default=200, type=int, help='hidden dimension of mlp')
    parser.add_argument('--learn_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
    parser.add_argument('--total_steps', default=2000, type=int, help='number of gradient steps')
    parser.add_argument('--noise', default=0.1, type=float, help='std of noise')
    # display
    parser.add_argument('--print_every', default=200, type=int, help='number of gradient steps between prints')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose?')
    parser.add_argument('--name', default='pend', type=str, help='only one option right now')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--save_dir', default=THIS_DIR, type=str, help='where to save the trained model')
    # model
    parser.add_argument('--model', default='hnn', type=str, help='used model.')
    parser.add_argument('--solver', default='dg', type=str, help='used solver.')
    parser.add_argument('--friction', default=False, action="store_true", help='use friction parameter')
    parser.set_defaults(feature=True)
    return parser.parse_args()


def train(args):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dtype = torch.get_default_dtype()
    torch.set_grad_enabled(False)

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # init model and optimizer
    model = dgnet.DGNet(args.input_dim, args.hidden_dim,
                        nonlinearity=args.nonlinearity, model=args.model, solver=args.solver)
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=1e-4)

    # arrange data
    t_span = 20
    length = 100
    dt = t_span / (length - 1)
    data = get_dataset(seed=args.seed, noise_std=args.noise, t_span=[0, t_span], timescale=length / t_span)
    train_x = torch.tensor(data['x'], requires_grad=True, device=device, dtype=dtype)
    test_x = torch.tensor(data['test_x'], requires_grad=True, device=device, dtype=dtype)
    train_dxdt = torch.tensor(data['dx'], device=device, dtype=dtype)
    test_dxdt = torch.tensor(data['test_dx'], device=device, dtype=dtype)

    input_dim = train_x.shape[-1]
    x_reshaped = train_x.view(-1, length, input_dim)
    x1 = x_reshaped[:, :-1].contiguous().view(-1, input_dim)
    x2 = x_reshaped[:, 1:].contiguous().view(-1, input_dim)
    dxdt = ((x2 - x1) / dt).detach()

    # vanilla train loop
    stats = {'train_loss': [], 'test_loss': []}
    for step in range(args.total_steps + 1):
        with torch.enable_grad():
            # train step
            dxdt_hat = model.discrete_time_derivative(x1, dt=dt, x2=x2)
            loss = L2_loss(dxdt, dxdt_hat)
        loss.backward()
        optim.step()
        optim.zero_grad()

        # run test data
        test_dxdt_hat = model.time_derivative(test_x)
        test_loss = L2_loss(test_dxdt, test_dxdt_hat)

        # logging
        stats['train_loss'].append(loss.item())
        stats['test_loss'].append(test_loss.item())
        if args.verbose and step % args.print_every == 0:
            print("step {}, train_loss {:.4e}, test_loss {:.4e}".format(step, loss.item(), test_loss.item()))

    train_dxdt_hat = model.time_derivative(train_x)
    train_dist = (train_dxdt - train_dxdt_hat)**2
    test_dxdt_hat = model.time_derivative(test_x)
    test_dist = (test_dxdt - test_dxdt_hat)**2
    print('Final train loss {:.4e}\nFinal test loss {:.4e}'
          .format(train_dist.mean().item(), test_dist.mean().item()))
    stats['final_train_loss'] = train_dist.mean().item()
    stats['final_test_loss'] = test_dist.mean().item()

    # energy error
    def integrate_models(x0=np.asarray([1, 0]), t_span=[0, 5], t_eval=None):
        from data import dynamics_fn
        import scipy.integrate
        rtol = 1e-12
        true_x = scipy.integrate.solve_ivp(fun=dynamics_fn, t_span=t_span, y0=x0, t_eval=t_eval, rtol=rtol)['y'].T
        # integrate along model vector field
        model_x = model.get_orbit(x0, t_eval, tol=rtol)
        return true_x, model_x

    def energy_loss(true_x, integrated_x):
        true_energy = (true_x**2).sum(1)
        integration_energy = (integrated_x**2).sum(1)
        return np.mean((true_energy - integration_energy)**2)

    t_span = [0, t_span]
    trials = 5 * 3
    t_eval = np.linspace(t_span[0], t_span[1], length)
    losses = {'model_energy': [], 'model_state': [], }

    true_orbits = []
    model_orbits = []
    for i in range(trials):
        x0 = np.random.rand(2) * 1.6 - .8  # randomly sample a starting px: \in(-2,2) and abs(px) > 0.2
        x0 += 0.2 * np.sign(x0) * np.ones_like(x0)
        true_x, model_x = integrate_models(x0=x0, t_span=t_span, t_eval=t_eval)
        true_orbits.append(true_x)
        model_orbits.append(model_x)
        losses['model_energy'] += [energy_loss(true_x, model_x)]
        losses['model_state'] += [((true_x - model_x)**2).mean()]
        print('{:.2f}% done'.format(100 * float(i) / (trials)), end='\r')

    stats['true_orbits'] = np.stack(true_orbits)
    stats['model_orbits'] = np.stack(model_orbits)
    stats['true_energies'] = (stats['true_orbits']**2).sum(-1)
    stats['model_energies'] = (stats['model_orbits']**2).sum(-1)

    losses = {k: np.array(v) for k, v in losses.items()}
    stats['energy_mse_mean'] = np.mean(losses['model_energy'])
    print("energy MSE {:.4e}".format(stats['energy_mse_mean']))
    stats['state_mse_mean'] = np.mean(losses['model_state'])
    print("state MSE {:.4e}".format(stats['state_mse_mean']))
    return model, stats


if __name__ == "__main__":
    args = get_args()

    # save
    os.makedirs(args.save_dir + '/results') if not os.path.exists(args.save_dir + '/results') else None
    label = ''
    label = label + '-{}-{}'.format(args.model, args.solver)
    label = label + '-friction' if args.friction else label
    label = label + '-seed{}'.format(args.seed)
    name = args.name
    result_path = '{}/results/dg-{}{}'.format(args.save_dir, name, label)
    path_tar = '{}.tar'.format(result_path)
    path_pkl = '{}.pkl'.format(result_path)
    path_txt = '{}.txt'.format(result_path)
    args.result_path = result_path

    if os.path.exists(path_txt):
        if args.noretry:
            exit()
        else:
            os.remove(path_txt)

    model, stats = train(args)
    torch.save(model.state_dict(), path_tar)
    to_pickle(stats, path_pkl)
    with open(path_txt, 'w') as of:
        print('#final_train_loss\tfinal_test_loss\tenergy_mse_mean\tstate_mse_mean', file=of)
        print(stats['final_train_loss'], stats['final_test_loss'], stats['energy_mse_mean'], stats['state_mse_mean'], sep='\t', file=of)
