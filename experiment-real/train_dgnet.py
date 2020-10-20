# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_default_dtype(torch.float32)
import argparse
import os
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
    # display
    parser.add_argument('--print_every', default=200, type=int, help='number of gradient steps between prints')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose?')
    parser.add_argument('--name', default='real', type=str, help='only one option right now')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--save_dir', default=THIS_DIR, type=str, help='where to save the trained model')
    # model
    parser.add_argument('--model', default='dgnet', type=str, help='used model.')
    parser.add_argument('--solver', default='implicit', type=str, help='used solver.')
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
                        nonlinearity=args.nonlinearity, friction=args.friction, model=args.model, solver=args.solver)
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=1e-5)

    # arrange data
    data = get_dataset('pend-real', args.save_dir)
    train_x = torch.tensor(data['x'], requires_grad=True, device=device, dtype=dtype)
    test_x = torch.tensor(data['test_x'], requires_grad=True, device=device, dtype=dtype)
    train_dxdt = torch.tensor(data['dx'], device=device, dtype=dtype)
    test_dxdt = torch.tensor(data['test_dx'], device=device, dtype=dtype)

    input_dim = train_x.shape[-1]
    x1 = train_x[:-1].detach()
    x2 = train_x[1:].detach()
    dxdt = train_dxdt[:-1].clone()
    dt = 1 / 6.

    # vanilla train loop
    stats = {'train_loss': [], 'test_loss': []}
    for step in range(args.total_steps + 1):
        with torch.enable_grad():
            # train step
            dxdt_hat = model.discrete_time_derivative(x1, dt=dt, x2=x2)
            loss = L2_loss(dxdt, dxdt_hat)
        optim.zero_grad()
        loss.backward()
        optim.step()

        # run validation
        if args.solver == 'implicit':
            # because it consumes too long time.
            test_loss = torch.tensor(float('nan'))
        else:
            test_dxdt_hat = model.discrete_time_derivative(test_x, dt=dt)
            test_loss = L2_loss(test_dxdt, test_dxdt_hat)

        # logging
        stats['train_loss'].append(loss.item())
        stats['test_loss'].append(test_loss.item())
        if args.verbose and step % args.print_every == 0:
            print("step {}, train_loss {:.4e}, test_loss {:.4e}".format(step, loss.item(), test_loss.item()))
            if args.friction:
                print("friction g =", model.g.detach().cpu().numpy())

    dxdt_hat = model.discrete_time_derivative(train_x, dt=dt)
    train_dist = (train_dxdt - dxdt_hat)**2
    test_dxdt_hat = model.discrete_time_derivative(test_x, dt=dt)
    test_dist = (test_dxdt - test_dxdt_hat)**2
    print('Final train loss {:.4e}\nFinal test loss {:.4e}'
          .format(train_dist.mean().item(), test_dist.mean().item()))
    stats['final_train_loss'] = train_dist.mean().item()
    stats['final_test_loss'] = test_dist.mean().item()

    # energy error
    from data import hamiltonian_fn
    t_eval = np.squeeze(data['test_t'] - data['test_t'].min())
    t_span = [t_eval.min(), t_eval.max()]
    x0 = test_x[0]
    true_orbits = test_x.detach().cpu().numpy()
    model_orbits = model.get_orbit(x0, t_eval=t_eval).detach().cpu().numpy()
    true_energies = np.stack([hamiltonian_fn(c) for c in true_orbits])
    model_energies = np.stack([hamiltonian_fn(c) for c in model_orbits])

    stats['true_orbits'] = true_orbits
    stats['model_orbits'] = model_orbits
    stats['true_energies'] = true_energies
    stats['model_energies'] = model_energies

    distance_energy = (true_energies - model_energies)**2
    distance_state = (true_orbits - model_orbits)**2
    stats['energy_mse_mean'] = np.mean(distance_energy)
    print("energy MSE {:.4e}".format(stats['energy_mse_mean']))
    stats['state_mse_mean'] = np.mean(distance_state)
    print("state MSE {:.4e}".format(stats['state_mse_mean']))
    return model, stats


if __name__ == "__main__":
    args = get_args()
    if args.model == 'node':
        args.friction = False

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
