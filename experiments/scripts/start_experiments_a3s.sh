#!/bin/bash

# export enviroment variables for slurm scripts

# results directory in experiments/results where the results are written
export RESULTS="aclust_compare_a3s_shallow"
# configuration for models
export CONFIG="configs/active_clustering_a3s.yaml"

# start the jobs for each dataset

sbatch run_active_clustering_segmentation_cpu.sh

sbatch run_active_clustering_webkb_cpu.sh

sbatch run_active_clustering_handwritten_cpu.sh

sbatch run_active_clustering_optdigits_cpu.sh

sbatch run_active_clustering_waveform_cpu_long.sh

sbatch run_active_clustering_har_cpu_long.sh

sbatch run_active_clustering_reuters_cpu_long.sh

sbatch run_active_clustering_usps_cpu_long.sh

sbatch run_active_clustering_pendigits_cpu_long.sh

sbatch run_active_clustering_bloodmnist_cpu_long.sh

sbatch run_active_clustering_mnist_cpu_long.sh

sbatch run_active_clustering_fashion_cpu_long.sh