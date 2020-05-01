import torch
import torch.nn as nn
import torch.nn.functional as F


class RBF(nn.Module):
    """
    Transforms incoming data using a given radial basis function:
    u_{i} = rbf(||x - c_{i}|| / s_{i})
    Arguments:
        in_features: size of each input sample
        out_features: size of each output sample
    Shape:
        - Input: (N, in_features) where N is an arbitrary batch size
        - Output: (N, out_features) where N is an arbitrary batch size
    Attributes:
        centres: the learnable centres of shape (out_features, in_features).
            The values are initialised from a standard normal distribution.
            Normalising inputs to have mean 0 and standard deviation 1 is
            recommended.

        sigmas: the learnable scaling factors of shape (out_features).
            The values are initialised as ones.

        basis_func: the radial basis function used to transform the scaled
            distances.
    """

    def __init__(self, input_size, output_size, basis_func):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.centres = nn.Parameter(torch.Tensor(output_size, input_size))
        self.sigmas = nn.Parameter(torch.Tensor(input_size))
        self.basis_func = basis_func
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.centres, 0, 1)
        nn.init.constant_(self.sigmas, 1)

    def forward(self, x):
        # Expand dimensions to match (batch_size, self.output_size, self.input_size)
        x = x.unsqueeze(1)
        c = self.centres.unsqueeze(0)
        s = self.sigmas.unsqueeze(0)
        sq_distances = ((x - c).pow(2) / torch.exp(s)).sum(-1)
        return self.basis_func(sq_distances)


class RBFNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, basis_func,
                 squash_func=None, output_bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rbf_layer = RBF(input_size, hidden_size, basis_func)
        self.linear_layer = nn.Linear(
            in_features=hidden_size, out_features=output_size, bias=output_bias)
        self.batch_mode = True
        self.squash_func = squash_func

    def forward(self, x):
        hidden = self.rbf_layer(x)
        out = self.linear_layer(hidden)
        # Apply the squashing function if provided
        out = self.squash_func(out) if self.squash_func else out
        # Remove the (possible) batch dimension if using for e.g. control
        out = out if self.batch_mode else out.view(-1)
        return out

    def set_batch_mode(self, batch_mode):
        self.batch_mode = batch_mode


class RandomPolicy(nn.Module):
    """Policy that samples an action uniformly at random from the action space."""

    def __init__(self, env):
        super().__init__()
        self.env = env

    def forward(self, x):
        return torch.FloatTensor(self.env.action_space.sample())


class MLPPolicy(nn.Module):
    """One-hidden layer multi-layer perceptron policy"""

    def __init__(self, input_size, output_size, hidden_size=100, output_bias=False,
                 squash_func = None):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.output_bias = output_bias
        self.squash_func = squash_func

        # Fully connected layers
        self.hidden = nn.Linear(in_features=self.input_size,
                                out_features=self.hidden_size,
                                bias=True)
        self.out = nn.Linear(in_features=self.hidden_size,
                             out_features=self.output_size,
                             bias=output_bias)

    def forward(self, x):
        hidden = torch.sigmoid(self.hidden(x))
        out = self.out(hidden)
        out = self.squash_func(out) if self.squash_func else out
        return out


def gaussian_rbf(sq_distances):
    phi = torch.exp(-0.5 * sq_distances)
    return phi


def sin_squash(unbounded_output, scale=10.):
    """Pass an unbounded output from a policy through a
    sin functin to bound it to [-scale, scale] range."""
    return scale*torch.sin(unbounded_output)


class SinSquash(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale
    
    def forward(self, x):
        return self.scale * torch.sin(x)
        
