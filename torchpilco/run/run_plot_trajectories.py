import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", type=str)
parser.add_argument("--iter", type=int)
parser.add_argument("--num_sim", type=int, default=10)
parser.add_argument("--save_path", type=str, default='runs/rewards_plot.png')


def main(args):
    true_traj_path= Path(args.log_dir) / f'step_{args.iter}_traj_true.txt'
    with true_traj_path.open() as f:
        true_trajectory = np.loadtxt(f)
    simulated_trajectories = []
    for sim in range(args.num_sim):
        traj_path= Path(args.log_dir) / f'step_{args.iter}_traj_sim_{sim}.txt'
        with traj_path.open() as f:
            simulated_trajectories.append(np.loadtxt(f))

    num_time_steps = true_trajectory.shape[0]
    
    colors = ['#FF69B4', '#0099Aa']
    with sns.axes_style("whitegrid"):
        fig, axes = plt.subplots(figsize=(12, 3.4), ncols=4)
        t = np.arange(num_time_steps)
        for i, ax in enumerate(axes):
            for sim in range(args.num_sim):
                ax.plot(t, simulated_trajectories[sim][:, i], linewidth=0.8, color=colors[0])
            ax.plot(t, true_trajectory[:, i], color=colors[1], linewidth=2.0)
            ax.set_xlabel('Time Step')
            ax.set_xlim(t[0], t[-1])
        axes[0].set_ylabel('Cart Position $x_c$')
        axes[1].set_ylabel('$\dot{x}_c$')
        axes[2].set_ylabel('Pendulum Angle $\\theta$')
        axes[3].set_ylabel('$\dot{\\theta}$')
    fig.tight_layout()
    fig.savefig(args.save_path, bbox_inches='tight', dpi=200)
    return

    
if __name__=='__main__':
    args = parser.parse_args()
    main(args)
    