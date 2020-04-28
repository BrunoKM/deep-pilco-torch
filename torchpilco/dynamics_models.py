import torch
import torch.nn as nn
import torch.nn.functional as F


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


class DynamicsNN(nn.Module):
    """
    A standard deep NN to model the transition dynamics.
    """
    def __init__(self, input_dim, output_dim, hidden_size=200, drop_prob=0.1):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.drop_prob = drop_prob

        # Fully connected layer
        self.fc1 = nn.Linear(in_features=self.input_dim,
                             out_features=self.hidden_size)
        self.fc2 = nn.Linear(in_features=self.hidden_size,
                             out_features=self.hidden_size)
        self.out = nn.Linear(in_features=self.hidden_size,
                             out_features=self.output_dim)
        # Dropout layers
        self.fc1_dropout = nn.Dropout(p=self.drop_prob)
        self.fc2_dropout = nn.Dropout(p=self.drop_prob)

    def forward(self, x):
        hidden1 = torch.sigmoid(self.fc1(x))
        hidden1 = self.fc1_dropout(hidden1)
        hidden2 = torch.sigmoid(self.fc2(hidden1))
        hidden2 = self.fc2_dropout(hidden2)
        output = self.out(hidden2)
        return output


class Ensemble(object):
    def __init__(self, models):
        assert type(models) is list
        self.models = models

    def __call__(self, x):
        return self.forward(x)

    def __len__(self):
        return len(self.models)

    def forward(self, x):
        y = []
        x_list = torch.chunk(x, len(self), dim=0)
        for model, x_chunk in zip(self.models, x_list):
            y.append(model(x_chunk))
        y = torch.cat(y, dim=0)
        return y

    @classmethod
    def load_from_savefile(cls, savedir_path, model_class, n_models=None):
        """Construct the ensemble object from a directory with several model savefiles"""
        models = []

        savedir_path = Path(savedir_path)
        save_files = [f for f in savedir_path.iterdir() if f.is_file() and f.suffix == '.pt']
        if n_models is not None:
            assert len(save_files) >= n_models
            save_files = save_files[:n_models]
        for save_file in save_files:
            models.append(load_model(model_class, save_file))
        return cls(models)

    def eval(self):
        for model in self.models:
            model.eval()

    def train(self):
        for model in self.models:
            model.train()

    def to(self, device):
        for model in self.models:
            model.to(device)
    
    def parameters(self):
        params = []
        for model in self.models:
            params.extend(list(model.parameters()))
        return params


def save_model(model, path):
    path = Path(path)  # Make sure path is a pathlib.Path object
    pathlib.Path(path.parent).mkdir(parents=True, exist_ok=True)  # Create directories if don't exist
    torch.save({
        'init_args': model.init_args,
        'model_state_dict': model.state_dict(),
    }, path)
    return


def load_model(model_class, path):
    checkpoint = torch.load(path)
    model = model_class(*checkpoint['init_args'])
    model.load_state_dict(checkpoint['model_state_dict'])
    return model