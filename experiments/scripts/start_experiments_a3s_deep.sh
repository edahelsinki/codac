#!/bin/bash

# export enviroment variables for slurm scripts

# results directory in experiments/results where the results are written
export RESULTS="aclust_compare_a3s_deep"
# configuration for models
export CONFIG="configs/active_clustering_a3s_deep.yaml"

# start the jobs for each dataset

sbatch run_active_clustering_segmentation.sh

sbatch run_active_clustering_webkb.sh

sbatch run_active_clustering_handwritten_long.sh

sbatch run_active_clustering_optdigits_long.sh

sbatch run_active_clustering_waveform_long.sh

sbatch run_active_clustering_har_long.sh

sbatch run_active_clustering_reuters_long.sh

sbatch run_active_clustering_usps_long.sh

sbatch run_active_clustering_pendigits_long.sh

sbatch run_active_clustering_bloodmnist_long.sh

sbatch run_active_clustering_mnist_long.sh

sbatch run_active_clustering_fashion_long.sh