import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchpilco.data import rollout


def eval_policy(env, policy, num_iter: int = 50, num_steps: int = 25, device=None):
    rewards_arr = np.zeros([num_iter])
    for i in range(num_iter):
        states, actions, rewards = rollout(env, policy, num_steps=num_steps, device=device)
        rewards_arr[i] = np.sum(rewards)
    return rewards_arr


def eval_policy_on_model(env, policy, dynamics_model, cost_function, num_particles: int = 10,
                         num_steps: int = 25, discount_factor=1.0, device=None,
                         moment_matching=False, mc_model=True):
    # Sample the initial state
    states = torch.FloatTensor([env.reset() for _ in range(num_particles)]).to(device)
    if mc_model:
        # Sample dynamics dropout masks (set batch_size=num_particles)
        dynamics_model.sample_new_mask(num_particles)

    total_cost = torch.mean(cost_function(states))

    for t in range(num_steps-1):
        actions = policy(states)
        # Concatenate particles and actions as inputs to Dynamics model
        state_action_tensor = torch.cat([states, actions], 1)
        # Get next states from the dynamics model
        state_deltas = dynamics_model(state_action_tensor)
        next_states = states + state_deltas

        # Moment matching
        if moment_matching:
            assert num_particles > 1
            mu = torch.mean(next_states, dim=0, keepdim=True)
            sigma = torch.std(next_states, dim=0, keepdim=True)
            # Standard normal noise for K particles
            z = torch.randn(num_particles, mu.size(1)).to(device)
            # Sample K new particles from a Gaussian (reparametrisation trick)
            next_states = mu + sigma * z
            # # Record mu and sigma
            # list_moments.append([mu, sigma])
        states = next_states
        total_cost += torch.mean(cost_function(states)) * discount_factor**t
    return -total_cost


def eval_mc_dynamics_model(dynamics_model,
                           testloader: DataLoader,
                           device=None,
                           log_interval: int = 100,
                           mc_model=True):
    #    summary_writer: SummaryWriter = None,
    dynamics_model.eval()
    criterion = nn.MSELoss()

    # start_time = time.time()
    total_loss = 0.
    for i, data in enumerate(testloader):
        # Get input batch
        x, y = data
        x, y = x.to(device), y.to(device)

        if mc_model:
            dynamics_model.sample_new_mask(batch_size=testloader.batch_size)

        # Forward pass
        outputs = dynamics_model(x)
        loss = criterion(outputs, y)
        total_loss += loss
    return total_loss / len(testloader)
