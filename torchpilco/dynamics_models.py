import torch
import torch.nn as nn


class MCDropoutDynamicsNN(nn.Module):
    """
    An MC-Dropout NN (variational approximation to a Bayesian NN) to model the transition dynamics.
    """

    def __init__(self, input_dim, output_dim, hidden_size=200,
                 drop_prob=0.1, batch_size=1, drop_input=True):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.drop_prob = drop_prob
        self.drop_input = drop_input  #  Whether to dropout input layer

        self.input_mask = None
        self.hidden1_mask = None
        self.hidden2_mask = None
        self.batch_size = batch_size

        # Fully connected layer
        self.fc1 = nn.Linear(in_features=self.input_dim,
                             out_features=self.hidden_size)
        self.fc2 = nn.Linear(in_features=self.hidden_size,
                             out_features=self.hidden_size)
        self.out = nn.Linear(in_features=self.hidden_size,
                             out_features=self.output_dim)

    def forward(self, x):
        if x.shape[0] != self.batch_size:
            raise ValueError(f'The input batch dimension is {x.shape[0]}, but the size of '
                             f'the sampled dropout mask is {self.batch_size}')
        x_in = x * self.input_mask if self.drop_input else x
        hidden1 = torch.sigmoid(self.fc1(x_in))
        # hidden1 = torch.sigmoid(self.fc1(x))
        hidden2 = torch.sigmoid(self.fc2(hidden1 * self.hidden1_mask))
        output = self.out(hidden2 * self.hidden2_mask)
        return output

    def sample_new_mask(self, batch_size=None):
        """Sample a new mask for MC-Dropout. Rather than sample the mask at each forward pass
        (as traditionally done in dropout), keep the dropped nodes fixed until this function is
        explicitely called.
        """
        device = self.get_param_device()
        if batch_size:
            self.batch_size = batch_size
        # Sample dropout random masks
        self.input_mask = torch.bernoulli(
            torch.ones(batch_size, self.input_dim) * (1 - self.drop_prob)).to(device)
        self.hidden1_mask = torch.bernoulli(
            torch.ones(batch_size, self.hidden_size) * (1 - self.drop_prob)).to(device)
        self.hidden2_mask = torch.bernoulli(
            torch.ones(batch_size, self.hidden_size) * (1 - self.drop_prob)).to(device)

    def get_param_device(self):
        return next(self.parameters()).device
