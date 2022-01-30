# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_default_dtype(torch.float64)
import argparse
import numpy as np

import os
import sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from hnn import HNN
from data import get_dataset
from utils import L2_loss, to_pickle, from_pickle
import dgnet


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--noretry', dest='noretry', action='store_true', help='not do a finished trial.')
    parser.add_argument('--load', default=False, action="store_true", help='load weight if given')
    # network, experiments
    parser.add_argument('--input_dim', default=1, type=int, help='dimensionality of input tensor')
    parser.add_argument('--hidden_dim', default=200, type=int, help='hidden dimension of mlp')
    parser.add_argument('--learn_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=200, type=int, help='batch_size')
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
    parser.add_argument('--total_steps', default=10000, type=int, help='number of gradient steps')
    # display
    parser.add_argument('--print_every', default=600, type=int, help='number of gradient steps between prints')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose?')
    parser.add_argument('--name', default='ch', type=str, help='only one option right now')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--save_dir', default=THIS_DIR, type=str, help='where to save the trained model')
    # model
    parser.add_argument('--model', default='hnn', type=str, help='used model.')
    parser.add_argument('--solver', default='dg', type=str, help='used solver.')
    parser.set_defaults(feature=True)
    return parser.parse_args()


def train(args):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    n_available_GPUs = torch.cuda.device_count()
    dtype = torch.get_default_dtype()
    torch.set_grad_enabled(False)

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # arrange data
    data = get_dataset(args.name, args.save_dir, verbose=True, device='cpu', test_split=0.1)
    train_u = torch.tensor(data['u'], requires_grad=True, device=device, dtype=dtype)
    test_u = torch.tensor(data['test_u'], requires_grad=True, device=device, dtype=dtype)

    train_dudt = torch.tensor(data['dudt'], device=device, dtype=dtype)
    test_dudt = torch.tensor(data['test_dudt'], device=device, dtype=dtype)

    t_eval = data['t_eval']
    dt = data['dt']
    M = test_u.shape[-1]

    train_shape_origin = train_u.shape
    test_shape_origin = test_u.shape
    u1 = train_u[:, :-1].contiguous().view(-1, 1, train_u.shape[-1])
    u2 = train_u[:, 1:].contiguous().view(-1, 1, train_u.shape[-1])
    dudt = ((u2 - u1) / dt).detach()

    train_u = train_u.view(-1, 1, train_u.shape[-1])
    test_u = test_u.view(-1, 1, test_u.shape[-1])
    train_dudt = train_dudt.view(-1, 1, train_dudt.shape[-1])
    test_dudt = test_dudt.view(-1, 1, test_dudt.shape[-1])

    # init model and optimizer
    alpha = 2 if args.name.startswith('ch') else 1
    model = dgnet.DGNetPDE1d(args.input_dim, args.hidden_dim,
                             nonlinearity=args.nonlinearity, model=args.model, solver=args.solver,
                             name=args.name, dx=data['dx'], alpha=alpha)
    print(model)
    model = model.to(device)
    stats = {'train_loss': [], 'test_loss': []}

    import glob
    files = glob.glob('{}.tar'.format(args.result_path))
    if len(files) > 0:
        f = files[0]
        path_tar = f
        model.load_state_dict(torch.load(path_tar, map_location=device))
        path_pkl = f.replace('.tar', '.pkl')
        stats = from_pickle(path_pkl)
        args.total_steps = 0
        print('Model successfully loaded from {}'.format(path_tar))

    if args.load:
        path_tar = '{}.tar'.format(args.result_path).replace('_long', '')
        model.load_state_dict(torch.load(path_tar, map_location=device))
        args.total_steps = 0
        print('Model successfully loaded from {}'.format(path_tar))

    optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=0)

    # vanilla train loop
    for step in range(args.total_steps):

        # train step
        idx = torch.randperm(u1.shape[0])[:args.batch_size]
        with torch.enable_grad():
            if n_available_GPUs > 1:
                dudt_hat = torch.nn.parallel.data_parallel(model, u1[idx], module_kwargs={'dt': dt, 'x2': u2[idx], 'func': 'discrete_time_derivative'})
            else:
                dudt_hat = model.discrete_time_derivative(u1[idx], dt=dt, x2=u2[idx])
            loss = L2_loss(dudt[idx], dudt_hat)
        optim.zero_grad()
        loss.backward()
        optim.step()

        # run test data
        test_idx = torch.randperm(test_u.shape[0])[:args.batch_size]
        test_dudt_hat = model.time_derivative(test_u[test_idx])
        test_loss = L2_loss(test_dudt[test_idx], test_dudt_hat)
        # logging
        stats['train_loss'].append(loss.item())
        stats['test_loss'].append(test_loss.item())
        if args.verbose and step % args.print_every == 0:
            print("step {}, train_loss {:.4e}, test_loss {:.4e}"
                  .format(step, loss.item(), test_loss.item()))
    if len(train_u) > 0:
        train_dudt_hat = torch.cat([model.time_derivative(train_u[idx:idx + args.batch_size]) for idx in range(0, len(train_u), args.batch_size)], dim=0)
        train_dist = (train_dudt - train_dudt_hat)**2
        test_dudt_hat = torch.cat([model.time_derivative(test_u[idx:idx + args.batch_size]) for idx in range(0, len(test_u), args.batch_size)], dim=0)
        test_dist = (test_dudt - test_dudt_hat)**2

        print('Final train loss {:.4e}\nFinal test loss {:.4e}'
              .format(train_dist.mean().item(), test_dist.mean().item()))
        stats['final_train_loss'] = train_dist.mean().item()
        stats['final_test_loss'] = test_dist.mean().item()
    else:
        stats['final_train_loss'] = 0.0
        stats['final_test_loss'] = 0.0

    # sequence generator
    os.makedirs('{}/results/'.format(args.save_dir), exist_ok=True)
    print('Generating test sequences')
    train_u = train_u.view(*train_shape_origin)
    test_u = test_u.view(*test_shape_origin)
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    test_u_truth = []
    test_u_model = []
    for idx in range(len(test_u)):
        print('Generating a sequence {}/{}'.format(idx, len(test_u)), end='\r')
        u_truth = test_u[idx].squeeze(1).detach().cpu().numpy()
        u_model = model.get_orbit(x0=test_u[idx, :1], t_eval=t_eval).squeeze(2).squeeze(1).detach().cpu().numpy()

        test_u_truth.append(u_truth)
        test_u_model.append(u_model)
        energy_truth = data['model'].get_energy(u_truth)
        energy_model = data['model'].get_energy(u_model)

        if args.model != 'node':
            energy_model_truth = model(torch.from_numpy(u_truth).unsqueeze(-2).to(device)).squeeze(2).squeeze(1).detach().cpu().numpy() * data['dx']
            energy_model_model = model(torch.from_numpy(u_model).unsqueeze(-2).to(device)).squeeze(2).squeeze(1).detach().cpu().numpy() * data['dx']

        mass_truth = u_truth.sum(-1)
        mass_model = u_model.sum(-1)

        if args.name.startswith('ch'):
            vmax = 1
            vmin = -1
        else:
            vmax = max(np.abs(u_truth).max(), np.abs(u_model).max())
            vmin = -vmax

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(9., 15.), facecolor='white')
        ax1.imshow(u_truth.T, interpolation='nearest', vmin=vmin, vmax=vmax, cmap='seismic')
        ax1.set_aspect('auto')
        ax1.set_yticks((-0.5, M - 0.5))
        ax1.set_yticklabels((0, 1))

        ax2.imshow(u_model.T, interpolation='nearest', vmin=vmin, vmax=vmax, cmap='seismic')
        ax2.set_aspect('auto')
        ax2.set_yticks((-0.5, M - 0.5))
        ax2.set_yticklabels((0, 1))

        ax3.plot([], [], color='white', label='energy')
        if args.model != 'node':
            ax3.plot(energy_model_truth - energy_model_truth[0], dashes=[2, 2], color='C0')
            ax3.plot(energy_model_model - energy_model_model[0], dashes=[2, 2], color='C1')
        ax3.plot(energy_truth - energy_truth[0], color='C0', label='ground truth')
        ax3.plot(energy_model - energy_model[0], color='C1', label=args.model)
        ax3.legend()

        ax4.plot([], [], color='white', label='mass')
        ax4.plot(mass_truth, color='C0')
        ax4.plot(mass_model, color='C1')
        ax4.set_xticks(t_eval[::len(t_eval) // 5] / dt)
        ax4.set_xticklabels(t_eval[::len(t_eval) // 5])
        ax4.set_xlabel('time')

        fig.savefig('{}_plot{:02d}.png'.format(args.result_path, idx))
        plt.close()

    test_u_truth = np.stack(test_u_truth, axis=0)[:, 1:]
    test_u_model = np.stack(test_u_model, axis=0)[:, 1:]
    energy_truth = data['model'].get_energy(test_u_truth)
    energy_model = data['model'].get_energy(test_u_model)

    print('energy MSE model', ((energy_truth - energy_model)**2).mean())
    stats['energy_mse_mean'] = ((energy_truth - energy_model)**2).mean()

    print('state MSE model', ((test_u_truth - test_u_model)**2).mean())
    stats['state_mse_mean'] = ((test_u_truth - test_u_model)**2).mean()

    stats['test_u_truth'] = test_u_truth
    stats['test_u_model'] = test_u_model
    stats['energy_truth'] = energy_truth
    stats['energy_model'] = energy_model

    if args.model != 'node':
        energy_model_truth = model(torch.from_numpy(test_u_truth).reshape(-1, 1, test_u_truth.shape[-1]).to(device)).detach().cpu().numpy().reshape(*test_u_truth.shape[:-1])
        energy_model_model = model(torch.from_numpy(test_u_model).reshape(-1, 1, test_u_model.shape[-1]).to(device)).detach().cpu().numpy().reshape(*test_u_model.shape[:-1])
        stats['energy_model_truth'] = energy_model_truth
        stats['energy_model_model'] = energy_model_model

    return model, stats


if __name__ == "__main__":
    args = get_args()
    # save
    os.makedirs(args.save_dir + '/results') if not os.path.exists(args.save_dir + '/results') else None
    label = ''
    label = label + '-{}-{}'.format(args.model, args.solver)
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
