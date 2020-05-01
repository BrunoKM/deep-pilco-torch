import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--log_dirs", type=str, nargs='+')
parser.add_argument("--save_path", type=str, default='runs/rewards_plot.png')
parser.add_argument("--labels", type=str, nargs='+')
parser.add_argument("--total_steps", type=int, default=100)


def main(args):
    rewards_list = []
    for log_dir in args.log_dirs:
        eval_rewards_path= Path(log_dir) / 'eval_rewards.txt'
        with eval_rewards_path.open() as f:
            rewards_list.append(-np.loadtxt(f) / 25.)
    
    colors = sns.color_palette('husl', len(args.log_dirs))
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(figsize=(9, 4))
        t = np.arange(args.total_steps) + 1
        for i, (rewards_arr, label) in enumerate(zip(rewards_list, args.labels)):
            if len(rewards_arr) < len(t):
                last_rewards = rewards_arr[-1]
                rewards_arr = np.concatenate((rewards_arr, np.tile(last_rewards, [len(t) - len(rewards_arr), 1])), axis=0)
            rewards_mean = np.mean(rewards_arr, axis=1)
            rewards_std = np.std(rewards_arr, axis=1)
            ax.fill_between(t, rewards_mean - rewards_std, rewards_mean + rewards_std, color=colors[i], alpha=0.3)
            ax.plot(t, rewards_mean, label=label, color=colors[i])
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Average Cost per Step')
        ax.set_xlim(t[0], t[-1])
        ax.set_xscale('log')
        plt.legend()
    fig.savefig(args.save_path, bbox_inches='tight', dpi=200)
    return

    
if __name__=='__main__':
    args = parser.parse_args()
    assert len(args.log_dirs) == len(args.labels)
    main(args)
    