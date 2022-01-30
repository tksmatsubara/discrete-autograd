from .modules import *
from .ode import DGNet


class DGNetPDE1d(DGNet):

    def __init__(self, input_dim=1, hidden_dim=200, nonlinearity='tanh', name='', dx=1., model='hnn', solver='dg', alpha=2):
        super(DGNetPDE1d, self).__init__(input_dim, hidden_dim, nonlinearity, model=model, solver=solver)
        output_dim = input_dim if self.model == 'node' else 1
        Act = get_decorated_module_by_name(nonlinearity)
        sequence = [
            PeriodicPad1d(1),
            Conv1d(input_dim, hidden_dim, kernel_size=3),
            Act(),
            Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            Act(),
        ]
        if model == 'node':
            sequence += [
                Conv1d(hidden_dim, output_dim, kernel_size=1),
            ]
        else:
            sequence += [
                Conv1d(hidden_dim, output_dim, kernel_size=1, bias=None),
                GlobalSummation1d(),
            ]
        self.net = Sequential(*sequence)

        self.dx = dx
        if alpha == 1:
            Ds = torch.tensor(
                [-1., 0., 1.],
                dtype=torch.get_default_dtype(),
            ).view(1, 1, 3) / (2 * self.dx)
        elif alpha == 2:
            Ds = torch.tensor(
                [1., -2., 1.],
                dtype=torch.get_default_dtype(),
            ).view(1, 1, 3) / self.dx / self.dx
        else:
            raise NotImplementedError(f'alpha={alpha} is not supported.')
        self.Ds = nn.Parameter(Ds, requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def time_derivative(self, x1, x2=None):
        if self.model in ['node']:
            return self.net(x1)
        else:
            grad = self.grad(x1, x2)
            grad_padded = periodic_pad1d(grad)
            return nn.functional.conv1d(grad_padded, self.Ds)
