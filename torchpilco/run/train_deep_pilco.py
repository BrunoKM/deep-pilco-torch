#!/usr/bin/env python3.8
import gym
import gym.envs.registration
import torch
import torch.nn.functional as F
import numpy as np
import os
from torchpilco.cartpole_swingup import CartPoleSwingUp, cartpole_cost_torch
from torchpilco.data import rollout, DynamicsDataBuffer, ScaledUpDataset, convert_trajectory_to_training
from torchpilco.dynamics_models import MCDropoutDynamicsNN
from torchpilco.policy_models import RBFNetwork, RandomPolicy, sin_squash, gaussian_rbf
from torchpilco.training import train_mc_dynamics_model, train_policy
from torchpilco.utils import plot_trajectory, plot_model_rollout_vs_true, create_summary_writer
from torchpilco.evaluation import eval_policy, eval_mc_dynamics_model, eval_policy_on_model
import seaborn as sns
import wandb

hyperparameter_defaults = dict(
    dropout=0.05,
    dynamics_hidden_size=200,
    dynamics_lr=5e-3,
    dynamics_weight_decay=1e-4,
    dynamics_batch_size=100,
    dynamics_num_iter=5000,
    policy_lr=5e-4,
    policy_num_iter=1000,
    num_steps_in_trial=25,
    policy_batch_size=10,
    num_policy_iter=1000,
    num_eval_trajectories=50,
    num_pilco_iter=50,
    discount_factor=1.0,
    buffer_size=10,
    policy_output_bias=0,
    squash_func='sin'
)

wandb.init(config=hyperparameter_defaults, project="model-based-rl-for-control")
config = wandb.config


def main(config):
    print(config)
    print('config squash func:', config.squash_func, '\tconfig policy_lr: ', config.policy_lr)
    writer = create_summary_writer('runs/deep_pilco')

    # Register the custom cartpole environment
    gym.envs.registration.register(id='CartPoleSwingUp-v0',
                                   entry_point='torchpilco.cartpole_swingup:CartPoleSwingUp')
    env = gym.make('CartPoleSwingUp-v0')

    # Set a random seed
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)

    # Dynamics
    dynamics_model = MCDropoutDynamicsNN(
        input_dim=env.observation_space.shape[0]+env.action_space.shape[0],
        output_dim=env.observation_space.shape[0],
        hidden_size=config.dynamics_hidden_size, drop_prob=config.dropout, drop_input=True)
    dynamics_optimizer = torch.optim.Adam(dynamics_model.parameters(
    ), lr=config.dynamics_lr, weight_decay=config.dynamics_weight_decay)
    wandb.watch(dynamics_model)

    # Policy
    if config.squash_func == 'sin':
        squash_func = lambda x: sin_squash(x, scale=10.0)
    elif config.squash_func == 'tanh':
        squash_func = lambda x: 10 * F.tanh(x)
    else:
        raise ValueError(f'Invalid squashing function: {config.squash_func}')
    rbf_policy = RBFNetwork(input_size=env.observation_space.shape[0], hidden_size=50,
                            output_size=env.action_space.shape[0], basis_func=gaussian_rbf,
                            squash_func=squash_func,
                            output_bias=bool(config.policy_output_bias))
    wandb.watch(rbf_policy)
    policy_optimizer = torch.optim.Adam(rbf_policy.parameters(), lr=config.policy_lr)
    # Define a random policy for initial experience
    rand_policy = RandomPolicy(env)

    data_buffer = DynamicsDataBuffer(capacity=config.num_steps_in_trial * config.buffer_size)
    test_data_buffer = DynamicsDataBuffer(
        capacity=config.num_steps_in_trial * config.num_eval_trajectories)
    test_dataloader = torch.utils.data.DataLoader(
        test_data_buffer, batch_size=config.dynamics_batch_size, shuffle=False, drop_last=True)
    # Generate eval data for dynamics model:
    for _ in range(config.num_eval_trajectories):
        states, actions, rewards = rollout(env, rand_policy, num_steps=config.num_steps_in_trial)
        test_data_buffer.push(*convert_trajectory_to_training(states, actions))

    # Initial experience:
    states, actions, rewards = rollout(env, rbf_policy, num_steps=config.num_steps_in_trial)
    writer.add_figure('sampled trajectory', plot_trajectory(states, actions, rewards)[0], 0)
    data_buffer.push(*convert_trajectory_to_training(states, actions))

    scaled_up_dataset = ScaledUpDataset(
        data_buffer, new_length=int(config.dynamics_batch_size*config.dynamics_num_iter))
    dataloader = torch.utils.data.DataLoader(
        scaled_up_dataset, batch_size=config.dynamics_batch_size, shuffle=True)

    for i in range(config.num_pilco_iter):
        # Evaluate dynamics model on a test-set collected with rand. policy
        dynamics_test_loss = eval_mc_dynamics_model(dynamics_model, test_dataloader)
        writer.add_scalar('dynamics_val_loss', dynamics_test_loss, i)
        wandb.log({'dynamics_val_loss': dynamics_test_loss})

        train_mc_dynamics_model(dynamics_model, dataloader, dynamics_optimizer,
                                summary_writer=writer,
                                start_step=i*config.dynamics_num_iter)
        train_policy(dynamics_model, rbf_policy, policy_optimizer, env,
                     cost_function=cartpole_cost_torch,
                     num_iter=config.num_policy_iter,
                     num_time_steps=config.num_steps_in_trial,
                     num_particles=config.policy_batch_size, moment_matching=True,
                     summary_writer=writer, start_step=i*config.num_policy_iter,
                     discount_factor=config.discount_factor)
        # Evaluate the policy
        eval_rewards = eval_policy(
            env, rbf_policy, num_iter=config.num_eval_trajectories,
            num_steps=config.num_steps_in_trial)
        eval_rewards_sim = eval_policy_on_model(
            env, rbf_policy, dynamics_model, cost_function=cartpole_cost_torch,
            num_particles = 50, num_steps=config.num_steps_in_trial, moment_matching=False).mean().data.cpu().numpy()

        writer.add_scalar('rewards/evaluation_real', eval_rewards.mean(), i+1)
        writer.add_scalar('rewards/evaluation_model', eval_rewards_sim, i+1)
        wandb.log({'eval_reward': eval_rewards.mean()})
        writer.add_histogram('rewards/evaluation_real_hist', eval_rewards, i+1)
        # Compare model trajectories to real trajectories
        writer.add_figure(
            'rollout trajectory vs true',plot_model_rollout_vs_true(
                env, rbf_policy, dynamics_model, cost_function=cartpole_cost_torch,
                num_model_runs=10, num_steps=config.num_steps_in_trial)[0], i+1)

        # Gather more experience
        states, actions, rewards = rollout(
            env, rbf_policy, num_steps=config.num_steps_in_trial)
        writer.add_figure('sampled trajectory', plot_trajectory(states, actions, rewards)[0], i+1)
        data_buffer.push(*convert_trajectory_to_training(states, actions))
        print(f'Iteration {i} complete')
    # Save model to wandb
    torch.save(dynamics_model.state_dict(), os.path.join(wandb.run.dir, 'dynamics_model.pt'))
    torch.save(rbf_policy.state_dict(), os.path.join(wandb.run.dir, 'policy_model.pt'))


if __name__ == '__main__':
    main(wandb.config)
