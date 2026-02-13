#!/bin/bash

# export enviroment variables for slurm scripts

# results directory in experiments/results where the results are written
export RESULTS="init_budget"
# configuration for models
export CONFIG="configs/ablation_init_budget.yaml"

# start the jobs for each dataset

sbatch run_active_clustering_bloodmnist.sh

sbatch run_active_clustering_mnist.sh

sbatch run_active_clustering_fashion.sh