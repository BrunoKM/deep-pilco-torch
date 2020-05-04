# Deep PILCO PyTorch Implementation

A reimplmentation of ["Improving PILCO with Bayesian Neural Network Dynamics Models"](http://mlg.eng.cam.ac.uk/yarin/PDFs/DeepPILCO.pdf) by Yarin Gal et al. in PyTorch.

## Average cost per iter. for the original MC Dropout and an Ensemble variant:
![alt text](https://github.com/BrunoKM/deep-pilco-torch/blob/master/figures/rewards-plot.png "Average Cost per Iteration")

The Deep Ensembles variant's hyperparameters have not been optimised, hence the comparatively poor performce. 

Even after an extensive hyperparameter search of the parameters not mentioned in the paper, the results obtained do not appear to quite match those obtained by original authors neither in [1] or [2].

# Run
#### Install dependencies
```
pip install requirements.txt
```
#### Install this repository in development mode
From the root of this repository (`.../deep-pilco-torch`):
```
pip install -e .
```
#### Run training
```
python torchpilco/run/train_deep_pilco.py
```
#### Make rewards plot
```
python run_plot_rewards.py --log_dirs {runs/deep_pilco_XX runs/deep_pilco_XX2} --labels {label-for-logdir-1 label-for-logdir-2} --save_path {where to save}
```
#### Make trajectory plots
```
python run_plot_trajectories.py --log_dir {runs/deep_pilco_XX} --iter {chosen iteration} --save_path {where to save}
```

## Plots of sample trajectories from the dynamics model
Sample trajectories using the trained policy at iteration 5:
![trajectories1](https://github.com/BrunoKM/deep-pilco-torch/blob/master/figures/traj_mc_1.pdf "Sample Trajectories")
At iteration 40:
![trajectories2](https://github.com/BrunoKM/deep-pilco-torch/blob/master/figures/traj_mc_2.pdf "Sample Trajectories")

[1] [Improving PILCO with Bayesian Neural Network Dynamics Models](http://mlg.eng.cam.ac.uk/yarin/PDFs/DeepPILCO.pdf), Yarin Gal and Rowan Thomas McAllister and Carl Edward Rasmussen

[2] [Uncertainty in Deep Learning](http://mlg.eng.cam.ac.uk/yarin/thesis/thesis.pdf)
