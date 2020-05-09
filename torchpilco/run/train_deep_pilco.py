import gym
import gym.envs.registration
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from torchpilco.cartpole_swingup import cartpole_cost_torch
from torchpilco.data import rollout, DynamicsDataBuffer, ScaledUpDataset
from torchpilco.data import convert_trajectory_to_training
from torchpilco.dynamics_models import MCDropoutDynamicsNN
from torchpilco.policy_models import RBFNetwork, RandomPolicy, MLPPolicy, sin_squash, gaussian_rbf
from torchpilco.training import train_dynamics_model, train_policy
from torchpilco.utils import plot_trajectory, plot_model_rollout_vs_true, new_run_directory
from torchpilco.evaluation import eval_policy, eval_dynamics_model, eval_policy_on_model
import wandb

hyperparameter_defaults = dict(
    dropout=0.05,
    dynamics_hidden_size=200,
    dynamics_lr=5e-5,
    dynamics_lr_gamma=1.0,  # Step Learning Rate schedule factor
    dynamics_weight_decay=1e-4,
    dynamics_batch_size=100,
    dynamics_num_iter=5000,
    policy_lr=7e-4,
    policy_lr_gamma=1.0,  # Step Learning Rate schedule factor
    policy_model='rbf',
    policy_batch_size=10,
    num_policy_iter=1000,
    num_eval_trajectories=50,
    num_pilco_iter=100,
    num_steps_in_trial=25,
    discount_factor=1.00,
    buffer_size=10,
    policy_output_bias=0,
    squash_func='tanh',
    random_seed=0,
    dropout_on_input=1,
)

wandb.init(config=hyperparameter_defaults, project="model-based-rl-for-control")
config = wandb.config


def main(config):
    print(config)
    run_dir = new_run_directory('runs/deep_pilco')
    writer = SummaryWriter(run_dir)

    # Register the custom cartpole environment
    gym.envs.registration.register(id='CartPoleSwingUp-v0',
                                   entry_point='torchpilco.cartpole_swingup:CartPoleSwingUp')
    env = gym.make('CartPoleSwingUp-v0')

    # Set a random seed
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    env.seed(config.random_seed)

    # Dynamics
    dynamics_model = MCDropoutDynamicsNN(
        input_dim=env.observation_space.shape[0]+env.action_space.shape[0],
        output_dim=env.observation_space.shape[0],
        hidden_size=config.dynamics_hidden_size, drop_prob=config.dropout,
        drop_input=bool(config.dropout_on_input))
    dynamics_optimizer = torch.optim.Adam(dynamics_model.parameters(
    ), lr=config.dynamics_lr, weight_decay=config.dynamics_weight_decay)
    dynamics_scheduler = torch.optim.lr_scheduler.StepLR(
        dynamics_optimizer, step_size=1, gamma=config.dynamics_lr_gamma)
    wandb.watch(dynamics_model)

    # Policy
    if config.squash_func == 'sin':
        def squash_func(x): return sin_squash(x, scale=10.0)
    elif config.squash_func == 'tanh':
        def squash_func(x): return 10 * torch.tanh(x)
    else:
        raise ValueError(f'Invalid squashing function: {config.squash_func}')

    if config.policy_model.lower() == 'rbf':
        policy_model = RBFNetwork(input_size=env.observation_space.shape[0], hidden_size=50,
                                  output_size=env.action_space.shape[0], basis_func=gaussian_rbf,
                                  squash_func=squash_func,
                                  output_bias=bool(config.policy_output_bias))
    elif config.policy_model.lower() == 'mlp':
        policy_model = MLPPolicy(input_size=env.observation_space.shape[0], hidden_size=100,
                                 output_size=env.action_space.shape[0],
                                 squash_func=squash_func,
                                 output_bias=bool(config.policy_output_bias))
    else:
        raise ValueError(f'Invalid Policy Model name: {config.policy_model}')
    wandb.watch(policy_model)
    policy_optimizer = torch.optim.Adam(policy_model.parameters(), lr=config.policy_lr)
    policy_scheduler = torch.optim.lr_scheduler.StepLR(
        policy_optimizer, step_size=1, gamma=config.policy_lr_gamma)
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
    states, actions, rewards = rollout(env, policy_model, num_steps=config.num_steps_in_trial)
    writer.add_figure('sampled trajectory', plot_trajectory(states, actions, rewards)[0], 0)
    data_buffer.push(*convert_trajectory_to_training(states, actions))

    scaled_up_dataset = ScaledUpDataset(
        data_buffer, new_length=int(config.dynamics_batch_size*config.dynamics_num_iter))
    dataloader = torch.utils.data.DataLoader(
        scaled_up_dataset, batch_size=config.dynamics_batch_size, shuffle=True)

    eval_rewards_list = []
    for i in range(config.num_pilco_iter):
        # Evaluate dynamics model on a test-set collected with rand. policy
        dynamics_test_loss = eval_dynamics_model(dynamics_model, test_dataloader)
        writer.add_scalar('dynamics_val_loss', dynamics_test_loss, i)
        wandb.log({'dynamics_val_loss': dynamics_test_loss})

        train_dynamics_model(dynamics_model, dataloader, dynamics_optimizer,
                             summary_writer=writer,
                             start_step=i*config.dynamics_num_iter)
        train_policy(dynamics_model, policy_model, policy_optimizer, env,
                     cost_function=cartpole_cost_torch,
                     num_iter=config.num_policy_iter,
                     num_time_steps=config.num_steps_in_trial,
                     num_particles=config.policy_batch_size, moment_matching=True,
                     summary_writer=writer, start_step=i*config.num_policy_iter,
                     discount_factor=config.discount_factor)
        # Evaluate the policy
        eval_rewards = eval_policy(
            env, policy_model, num_iter=config.num_eval_trajectories,
            num_steps=config.num_steps_in_trial)
        eval_rewards_list.append(eval_rewards)

        eval_rewards_sim = eval_policy_on_model(
            env, policy_model, dynamics_model, cost_function=cartpole_cost_torch,
            num_particles=50, num_steps=config.num_steps_in_trial, moment_matching=False).mean().data.cpu().numpy()

        writer.add_scalar('rewards/evaluation_real', eval_rewards.mean(), i+1)
        writer.add_scalar('rewards/evaluation_model', eval_rewards_sim, i+1)
        wandb.log({'eval_reward': eval_rewards.mean()})
        writer.add_histogram('rewards/evaluation_real_hist', eval_rewards, i+1)
        # Compare model trajectories to real trajectories
        writer.add_figure(
            'rollout trajectory vs true', plot_model_rollout_vs_true(
                env, policy_model, dynamics_model, cost_function=cartpole_cost_torch,
                num_model_runs=10, num_steps=config.num_steps_in_trial,
                log_dir=run_dir, log_name=f'step_{i}_traj')[0], i+1)

        # Gather more experience
        states, actions, rewards = rollout(
            env, policy_model, num_steps=config.num_steps_in_trial)
        writer.add_figure('sampled trajectory', plot_trajectory(states, actions, rewards)[0], i+1)
        data_buffer.push(*convert_trajectory_to_training(states, actions))
        # Adjust learning rates
        dynamics_scheduler.step()
        policy_scheduler.step()
        print(f'Iteration {i} complete')
    # Save model to wandb
    torch.save(dynamics_model.state_dict(), os.path.join(wandb.run.dir, 'dynamics_model.pt'))
    torch.save(policy_model.state_dict(), os.path.join(wandb.run.dir, 'policy_model.pt'))

    eval_rewards_path = run_dir / f'eval_rewards.txt'
    with eval_rewards_path.open('w') as f:
        np.savetxt(f, np.stack(eval_rewards_list, axis=0))


if __name__ == '__main__':
    main(wandb.config)
