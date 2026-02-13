#!/bin/bash

# export enviroment variables for slurm scripts

# results directory in experiments/results where the results are written
export RESULTS="constrained_clustering"

sbatch run_constrait_clustering_mnist.sh
sbatch run_constrait_clustering_fashion.sh
sbatch run_constrait_clustering_bloodmnist.sh