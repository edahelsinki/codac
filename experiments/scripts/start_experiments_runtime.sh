#!/bin/bash

# export enviroment variables for slurm scripts

# results directory in experiments/results where the results are written
export RESULTS="runtime_codac"
# configuration for models
export CONFIG="configs/active_clustering_codac.yaml"

# start the jobs for each dataset

sbatch run_runtime_fashion.sh