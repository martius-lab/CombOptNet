# CombOptNet: Fit the Right NP-Hard Problem by Learning Integer Programming Constraints

![Architecture overview](media/arch_overview.png)


This repository contains PyTorch implementation of the paper
[CombOptNet: Fit the Right NP-Hard Problem by Learning Integer Programming Constraints](https://arxiv.org/abs/2105.02343)

The main branch contains the training code for replicating the experimental results.
For the results from the graph matching experiment please refer to the comboptnet branch in the 
[BB-GM](https://github.com/martius-lab/blackbox-deep-graph-matching) repository.
The independent underlying implementation of CombOptNet is installed from the comboptnet-module branch,
a useful handler for initializing a learnable set of constraints is installed from the constraint-handler branch.


## Replicating the experimental results
#### Installation
1) Run `pipenv install` (at your own risk with `--skip-lock` to save some time).
2) Obtain a gurobi [license](https://www.gurobi.com/documentation/9.1/quickstart_mac/obtaining_a_grb_license.html) and download/set it.
3) Download and extract the [datasets](https://edmond.mpdl.mpg.de/imeji/collection/Z_abYaB4ggQTS_G0?q=).

#### Usage
For `[experiment] = knapsack` or `[experiment] = static_constraints`:
1) Set the `base_dataset_path` parameter in `experiments/[experiment]/base.yaml`.
2) In case of static constraints: set the `dataset_specification` parameter in `experiments/static_constraints/base.yaml`
3) Run `python3 main.py experiments/[experiment]/[method].yaml`.


## Using CombOptNet
Install the differentiable ILP solver from the comboptnet-module branch using pip:

``python3 -m pip install git+https://github.com/martius-lab/CombOptNet@comboptnet-module``

Optionally install the handler maintaining a leanrable set of constraints from the constraint-handler branch using pip:

``python3 -m pip install git+https://github.com/martius-lab/CombOptNet@constraint-handler``