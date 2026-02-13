#!/bin/bash

# export enviroment variables for slurm scripts

# results directory in experiments/results where the results are written
export RESULTS="guess_k"
# configuration for models
export CONFIG="configs/active_k_estim.yaml"

# start the jobs for each dataset

sbatch run_active_clustering_segmentation.sh

sbatch run_active_clustering_webkb.sh

sbatch run_active_clustering_handwritten.sh

sbatch run_active_clustering_optdigits.sh

sbatch run_active_clustering_waveform.sh

sbatch run_active_clustering_har.sh

sbatch run_active_clustering_usps.sh

sbatch run_active_clustering_pendigits.sh

sbatch run_active_clustering_bloodmnist.sh

sbatch run_active_clustering_reuters.sh

sbatch run_active_clustering_mnist.sh

sbatch run_active_clustering_fashion.sh