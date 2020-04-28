import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchpilco.dynamics_models import MCDropoutDynamicsNN


def train_mc_dynamics_model(dynamics_model,
                            trainloader: DataLoader,
                            dynamics_optimizer: optim.Optimizer,
                            device=None,
                            log_interval: int = 100,
                            summary_writer: SummaryWriter = None,
                            start_step: int = 0):
    dynamics_model.train()
    criterion = nn.MSELoss()

    # start_time = time.time()
    for i, data in enumerate(trainloader):
        # Get input batch
        x, y = data
        x, y = x.to(device), y.to(device)

        dynamics_optimizer.zero_grad()
        dynamics_model.sample_new_mask(batch_size=trainloader.batch_size)

        # Forward pass
        outputs = dynamics_model(x)

        loss = criterion(outputs, y)
        loss.backward()
        dynamics_optimizer.step()
        # Log training statistics
        if i % log_interval == 0:
            # wandb.log({"loss": loss})
            if summary_writer:
                summary_writer.add_scalar('dynamics loss', loss, start_step + i)
            print(f'Step: {i} \tLoss: {loss.cpu()}')


def train_policy(dynamics_model: MCDropoutDynamicsNN,
                 policy_model,
                 policy_optimizer,
                 env,
                 cost_function,
                 num_iter: int,
                 num_particles=10,
                 num_time_steps=25,
                 moment_matching=True,
                 discount_factor=1.0,
                 device=None,
                 log_interval: int = 10,
                 summary_writer: SummaryWriter = None,
                 start_step: int = 0):
    dynamics_model.eval()
    policy_model.train()
    for i in range(num_iter):
        # Sample the initial state
        states = torch.FloatTensor([env.reset() for _ in range(num_particles)]).to(device)

        # Sample dynamics dropout masks (set batch_size=num_particles)
        dynamics_model.sample_new_mask(num_particles)

        total_cost = torch.mean(cost_function(states))

        for t in range(num_time_steps-1):
            actions = policy_model(states)
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

        # Optimize policy
        policy_optimizer.zero_grad()
        total_cost.backward()
        # total_cost.backward(retain_graph=True)
        policy_optimizer.step()
        # Logging and monitoring
        if i % log_interval == 0:
            # wandb.log({'cost': total_cost})
            if summary_writer:
                summary_writer.add_scalar('policy cost', total_cost, start_step + i)
                log_gradient_hist(policy_model, summary_writer,
                                  step=start_step + i, hist_name='policy_grads')
            print(f"Step\t{i}\tLoss:\t{total_cost}")


def log_gradient_hist(model: nn.Module, writer: SummaryWriter,
                      step: int, hist_name='grads'):
    for name, param in model.named_parameters():
        writer.add_histogram(hist_name + '/' + name, param, step)

# """ Use cost
# def optimize_policy(list_costs, dynamics, policy, policy_optimizer):
#     policy_optimizer.zero_grad()

#     #for c in list_costs[:-1]:
#     #    c.backward(retain_graph=True)
#     #list_costs[-1].backward()

#     J = torch.sum(torch.cat(list_costs))
#     J.backward(retain_graph=True)

#     policy_optimizer.step()
# #"""

# def _grad_norm(net):
#     norm = 0.0
#     for param in net.parameters():
#         if param.grad is not None:
#             norm += np.linalg.norm(param.grad.data.cpu().numpy())

#     return norm

# def _clone_net(net, requires_grad=False):
#     if type(net) is LinearPolicy:
#         cloned = type(net)(net.env)
#     else:
#         cloned = type(net)(net.env, net.hidden_size)

#     for cloned_param, net_param in zip(cloned.parameters(), net.parameters()):
#         cloned_param.data = net_param.data.clone()
#         cloned_param.requires_grad = requires_grad

#     return cloned