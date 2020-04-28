import torch
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def rollout(env, policy, num_steps=1, device=None):
    """Generate one trajectory with NN or System dynamics, return transitions"""
    cur_state = env.reset()
    
    states = [cur_state]
    actions = []
    rewards = [-env.compute_cost()]
    for _ in range(num_steps-1):
        # Convert to FloatTensor feedable into a Torch model
        cur_state = torch.FloatTensor(cur_state).unsqueeze(0).to(device)
        action = torch.flatten(policy(cur_state))  # Ensure ndims=1
        action = action.data.cpu().numpy()
        next_state, reward, *_ = env.step(action)
        # Record data
        actions.append(action)
        states.append(next_state)
        rewards.append(reward)
        
        cur_state = next_state
    # Convert to numpy arrays
    states, actions, rewards = tuple(map(lambda l: np.stack(l, axis=0),
                                         (states, actions, rewards)))
    return states, actions, rewards


def dynamics_model_rollout(env, policy, dynamics_model, cost_function, num_particles: int = 10,
                           num_steps: int = 25, init_states=None, device=None, moment_matching:bool =False,
                           mc_model=True):
    dynamics_model.eval()
    # Sample the initial state
    if init_states is None:
        cur_states = torch.FloatTensor([env.reset() for _ in range(num_particles)]).to(device)
    else:
        cur_states = torch.FloatTensor(init_states)
        assert cur_states.shape[0] == num_particles
    if mc_model:
        # Sample dynamics dropout masks (set batch_size=num_particles)
        dynamics_model.sample_new_mask(num_particles)

    states = [cur_states.data.cpu().numpy()]
    actions = []
    rewards = [-cost_function(cur_states).data.cpu().numpy()]
    for t in range(num_steps-1):
        action = policy(cur_states)
        # Concatenate particles and actions as inputs to Dynamics model
        state_action_tensor = torch.cat([cur_states, action], 1)
        # Get next states from the dynamics model
        state_deltas = dynamics_model(state_action_tensor)
        next_states = cur_states + state_deltas

        # Moment matching
        if moment_matching:
            assert num_particles > 1
            mu = torch.mean(next_states, dim=0, keepdim=True)
            sigma = torch.std(next_states, dim=0, keepdim=True)
            # Standard normal noise for particles
            z = torch.randn(num_particles, mu.size(1)).to(device)
            # Sample K new particles from a Gaussian 
            next_states = mu + sigma * z
        cur_states = next_states
        states.append(cur_states.data.cpu().numpy())
        actions.append(action.data.cpu().numpy())
        rewards.append(-cost_function(cur_states).data.cpu().numpy())
    states, actions, rewards = tuple(map(lambda l: np.stack(l, axis=1),
                                         (states, actions, rewards)))
    return states, actions, rewards


def convert_trajectory_to_training(states, actions):
    """Convert a sequence of states and actions into training data (X, Y) for a
    dynamics model.
    
    The training data has input X - which is the concatenated state and action
    taken in that state - and output Y, which is the difference between the next state
    and the current state.
    
    Args:
        states (np.ndarray): numpy array of shape [N, state_dim]
        actions (np.ndarray): numpy array of shape [N - 1, action_dim]
    Returns:
        (x, y): where x is a numpy array of shape [N, state_dim + action_dim] and
            y is a numpy array of shape [N, state_dim]
    """
    assert states.shape[0] == actions.shape[0] + 1
    x = np.concatenate((states[:-1], actions), axis=1)
    y = states[1:] - states[:-1]
    return x, y


class DynamicsDataBuffer(data.Dataset):
    def __init__(self, capacity=10):
        self.data = []
        self.capacity = capacity
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.FloatTensor(x), torch.FloatTensor(y)

    def __repr__(self):
        return f'Dynamics Data Buffer with {len(self.data)} / {self.capacity} elements.\n'

    def push(self, x: np.ndarray, y: np.ndarray):
        if x.ndim == 1:
            # In case this is a single datapoint, ensure ndims == 2 (add batch dimension)
            assert y.ndim == 1
            x = x[None, :]
            y = y[None, :]
        for i in range(x.shape[0]):
            self.data.append((x[i], y[i]))
        # Ensure capacity isn't exceeded
        if len(self.data) > self.capacity:
            del self.data[:len(self.data) - self.capacity]


class ScaledUpDataset(data.Dataset):
    """Helper class for extending the dataset (repeating it).
    This is useful when it is known a certain number of examples is required (e.g. to fill
    a batch), but the amount of data might not be sufficient.
    """
    def __init__(self, dataset, new_length):
        self.dataset = dataset
        self.length = new_length
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if idx > self.length:
            raise IndexError(f'Index {idx} out of bounds for dataset of size{self.length}')
        if idx < 0:
            raise IndexError("Index must be non-negative")
        return self.dataset[idx % len(self.dataset)]


# def dynamics_model_rollout():
#     # Intial state
#     if init_particle is not None:
#         init_state = init_particle
#         env.reset()
#         # Unwrap accesses the underlying class of a gym environment
#         env.unwrapped.state = init_particle
#     else:
#         init_state = env.reset()
    
#     transitions = []
#     for _ in range(num_steps):
#         # Convert to FloatTensor, Variable and send to GPU
#         init_state = Variable(torch.FloatTensor(init_state).unsqueeze(0)).to(device)
#         # Select an action by policy
#         action = policy(init_state)
#         # Take action via NN/System dynamics
#         if mode == 'System':
#             s_next, _, _, _ = env.step(action.data.cpu().numpy())
#         elif mode == 'NN':
#             state_action = torch.cat([init_state, action.unsqueeze(0)], 1)
#             s_next = dynamics(state_action).data.cpu().numpy()[0]
#         else:
#             raise ValueError('The value of mode must be either NN or System. ')
        
#         # Record data
#         transitions.append(np.concatenate([init_state.data.cpu().numpy()[0],
#                                            action.data.cpu().numpy(), s_next]))
        
#         # Update s as s_next for recording next transition
#         init_state = s_next
        
#     return np.array(transitions)

