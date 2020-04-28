import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
from torchpilco.data import rollout, dynamics_model_rollout
from pathlib import Path
from typing import Union

sns.set()


def execute_torch_policy(observation, policy, device=None):
    """Wrapper around calling a policy that handles converting observation
    to torch and sending to device, and then converting output of policy
    back into numpy.
    """
    observation_tensor = torch.Tensor(observation).to(device).unsqueeze(0)
    return policy(observation_tensor).detach().cpu().numpy()


def plot_databuffer(data, ylim=None):
    fig, ax = plt.subplots(1, 1)

    for i in range(len(data.buffer)):
        tau = data.buffer[i]

        s = np.concatenate([tau[:, 2], [tau[-1, -2]]])

        if i == len(data.buffer) - 1:  # latest trajectory in red
            ax.plot(np.arange(0, s.shape[0]), s, 'red')
        else:
            ax.plot(np.arange(0, s.shape[0]), s, 'blue')

    # Figure settings
    ax.grid()
    ax.set_title('Trajectories in data buffer')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Angle (in radians)')
    if ylim is not None:
        ax.set_ylim(ylim)
    plt.close()  # Close figure to save memory

    return fig


def new_run_directory(path: Union[str, Path]):
    path = Path(path)
    run_name = path.name
    previous_runs = path.parent.glob(f'{run_name}*')
    # Remove run_name from run directories
    prev_run_nums = map(lambda prev_path: prev_path.name[len(run_name):], previous_runs)
    # Remove those runs that do not have a number at the end
    prev_run_nums = filter(lambda prev_run_num: prev_run_num.isdigit(), prev_run_nums)
    # Convert to int and sort
    prev_run_nums = sorted(map(lambda prev_run_num: int(prev_run_num), prev_run_nums))
    # If there are any previous runs
    if prev_run_nums:
        new_run_num = int(prev_run_nums[-1]) + 1
    else:
        new_run_num = 1
    new_run_dir = Path(str(path) + str(new_run_num))
    return new_run_dir


def create_summary_writer(path: Union[str, Path]):
    """An utility for creating tensorboard SummaryWriter
    objects without overriding previous runs. It takes a given name
    and appends a number to it to make it distinct from previous runs if such exist.
    """
    new_run_dir = new_run_directory(path)
    return SummaryWriter(new_run_dir)


def plot_trajectory(states, actions, rewards, state_names=None):
    with sns.axes_style("whitegrid"):
        fig, axes = plt.subplots(ncols=3, figsize=(10, 3), sharex=True)
        # Plot state
        t = np.arange(states.shape[0])
        for i in range(states.shape[1]):
            label = state_names[i] if state_names else f'State var. {i+1}'
            axes[0].plot(t, states[:, i], label=label)
        axes[0].legend()
        axes[0].set_xlim(0, t[-1])

        # Plot rewards
        axes[2].plot(t, rewards, label='Reward')

        # Plot actions
        if actions.shape[0] != states.shape[0]:
            assert actions.shape[0] == states.shape[0] - 1
            t = np.arange(actions.shape[0])
        if actions.shape[1] == 1:
            axes[1].plot(t, actions[:, 0], label='Action')
        else:
            for i in range(actions.shape[1]):
                label = f'Action {i+1}'
                axes[1].plot(t, actions[:, i], label=label)
        axes[0].legend()
        axes[1].legend()
        axes[2].legend()
    return fig, axes


def plot_model_rollout_vs_true(env, policy, dynamics_model, cost_function, num_model_runs: int = 10,
                           num_steps: int = 25, device=None, moment_matching:bool =False,
                           mc_model=True, log_dir=None, log_name='rollout'):
    true_states, true_actions, true_rewards = rollout(env, policy, num_steps=num_steps, device=device)
    init_state = true_states[0, :]
    init_states = np.repeat(init_state[None, :], num_model_runs, axis=0)

    model_states, model_actions, model_rewards = dynamics_model_rollout(
        env, policy=policy, dynamics_model=dynamics_model, cost_function=cost_function,
        num_particles= num_model_runs, num_steps=num_steps, init_states=init_states, device=device, moment_matching=False, mc_model=mc_model)
    if log_dir:
        # Save dynamics model predicted trajectories
        for i in range(model_states.shape[0]):
            f_path = Path(log_dir) / (log_name + f'_sim_{i}.txt')
            with f_path.open('w') as f:
                np.savetxt(f, model_states[i])
        # Save true trajectory
        f_path = Path(log_dir) / (log_name + '_true.txt')
        with f_path.open('w') as f:
            np.savetxt(f, true_states)

    with sns.axes_style("whitegrid"):
        fig, axes = plt.subplots(ncols=true_states.shape[1], figsize=(10, 3), sharex=True)
        t = np.arange(true_states.shape[0])
        # Plot simulated states
        for sim in range(model_states.shape[0]):
            for i in range(true_states.shape[1]):
                axes[i].plot(t, model_states[sim, :, i], color='green', alpha=0.5)
        
        # Plot true states
        for i in range(true_states.shape[1]):
            axes[i].plot(t, true_states[:, i], color='blue')
            axes[i].set_xlim(0, t[-1])
            axes[i].set_xlabel('Time Step')
            axes[i].set_xlabel(f'State Variable {i+1}')
    return fig, axes
