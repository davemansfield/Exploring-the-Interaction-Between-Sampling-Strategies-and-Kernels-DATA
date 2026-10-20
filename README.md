# Exploring the Interaction Between Sampling Strategies and Kernels in Gaussian Process Regression for Autonomous Environmental Sensing

Code and data accompanying the paper:

> D. Mansfield, H. Ghadiri and A. Montazeri, *Exploring the Interaction Between Sampling Strategies and Kernels in Gaussian Process Regression for Autonomous Environmental Sensing*.

The study evaluates four GPR kernels (squared exponential, Matérn 3/2, Matérn 5/2 and piecewise polynomial, each in isotropic and ARD form) paired with four sampling strategies (random, greedy, infotaxis and GP-UCB) across three environments, under the computational and movement constraints of a mobile sensing platform. It also proposes an online kernel-switching framework based on negative marginal log-likelihood (NMLL) comparison.

## Requirements

- MATLAB R2023b (R2022b or newer is required for `rmse`)
- Statistics and Machine Learning Toolbox (`rangesearch`, `mvnrnd`)
- [GPML MATLAB Toolbox v4.2](https://gaussianprocess.org/gpml/code/matlab/doc/) (Rasmussen & Nickisch), downloaded separately

## Setup

From the repository root in MATLAB:

```matlab
setup('/path/to/gpml-matlab')   % adds the GPML core folders and this repository to the path
```

If GPML is already on your path, call `setup` with no arguments.

## Repository structure

```text
setup.m                  Adds the repository and GPML to the MATLAB path
functions/               Shared functions
  cov_reset.m              Kernel definitions and initial hyperparameters (Table 1)
  get_full_k_matrix.m      Covariance used to generate Environment 2
  GPUCBacqFunc.m           GP-UCB acquisition function (Eq. 12)
run_experiments/
  experiment_1/          Experiment 1: static vs iterative learning (Section 2.4.2)
  experiment_2/          Experiment 2: kernel-sampling interaction (Section 2.4.3)
                         and sensitivity analysis (Section 4.4)
  experiment_3/          Adaptive kernel switching (Section 4.6)
```

### File naming

Experiment scripts are named `Ex<experiment>_<environment>_<method>.m`, where the environment is:

| Name       | Environment | Description                                          |
|------------|-------------|------------------------------------------------------|
| `math`     | 1           | Deterministic function sin(x2) + x1                  |
| `GP`       | 2           | Gaussian process sample, SE ARD, θ = log(1.5, 1.0, 1.9) |
| `realdata` | 3           | Intel Berkeley Research Lab temperature data         |

In Experiment 1, `set` denotes learning from a static dataset and `it` from iteratively collected samples. In Experiments 2 and 3, the method is the sampling algorithm: `random`, `greedy`, `entropy` (infotaxis, Algorithm 3) or `gpucb` (GP-UCB, Algorithm 2).

### Kernel indices

Kernels are selected by index `n` in `cov_reset(n)`:

| n | Kernel     | n | Kernel     |
|---|------------|---|------------|
| 1 | SE iso     | 5 | Matérn 5/2 iso |
| 2 | SE ARD     | 6 | Matérn 5/2 ARD |
| 3 | Matérn 3/2 iso | 7 | PP iso     |
| 4 | Matérn 3/2 ARD | 8 | PP ARD     |

## Data

### Results

Results are stored as MATLAB variables named

```text
save_<metric>_cov<n>_env<e>_<algorithm>
```

where `<metric>` is one of `RMSE`, `NMLL`, `timeopt` (hyperparameter optimisation time), `timestep` (acquisition time), `ls1` and `ls2` (first two log-hyperparameters); `<n>` is the kernel index above; `<e>` is the environment; and `<algorithm>` is `random`, `greedy`, `entropy` (infotaxis) or `gpucb`. Each variable is an `N_steps × N_run` matrix (100 × 50 in the paper).

### Environment 3 source data

Environment 3 uses the [Intel Berkeley Research Lab dataset](https://db.csail.mit.edu/labdata/labdata.html) (Bodik et al., 2004). The `realdata` scripts expect a table named `data` in the workspace with columns `epoch`, `x`, `y` and `temp` (sensor readings joined with sensor locations). Readings at `epoch = 300` are interpolated onto a 50 × 50 grid over [6, 30] × [6, 30] using natural-neighbour interpolation (`griddata`), as described in Section 2.4.1.

## Reproducing the paper

| Paper section | Results | Scripts |
|---|---|---|
| Experiment 1: static vs iterative learning (Section 3.1) | Figures 5, 6, 20 | `experiment_1/Ex1_<env>_set.m`, `experiment_1/Ex1_<env>_it.m` |
| Experiment 2: kernel-sampling interaction (Section 3.2) | Figures 7–16, 19, 21, 22; Tables 2–5, 7 | `experiment_2/Ex2_<env>_<algorithm>.m` |
| Sensitivity analysis (Section 4.4) | Figure 17 | `experiment_2/EX2_GP_gpucb_abalation.m` |
| Adaptive kernel switching (Section 4.6) | Figure 18, Table 6 | `experiment_3/EX3_GP_<algorithm>.m` |

All experiments in the paper use 50 runs of 100 samples, 20 L-BFGS iterations per sampling step, a movement radius r = 1 on a 50 × 50 grid over [−5, 5]², and observation noise N(0, 0.1²).

## License

The code in this repository is released under the [MIT License](LICENSE). The GPML toolbox and the Intel Berkeley Research Lab dataset are distributed separately under their own terms.

## Acknowledgements

This work was supported by the Engineering and Physical Sciences Research Council (EPSRC), grant number EP/T518037/1, and by the Lancaster Environment Centre (LEC).
