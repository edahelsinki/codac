#!/bin/bash

# export enviroment variables for slurm scripts

# results directory in experiments/results where the results are written
export RESULTS="aclust_compare_shallow"
# configuration for models
export CONFIG="configs/active_clustering_shallow.yaml"

# start the jobs for each dataset

sbatch run_active_clustering_segmentation_cpu.sh

sbatch run_active_clustering_webkb_cpu.sh

sbatch run_active_clustering_handwritten_cpu.sh

sbatch run_active_clustering_optdigits_cpu.sh

sbatch run_active_clustering_waveform_cpu.sh

sbatch run_active_clustering_har_cpu.sh

sbatch run_active_clustering_reuters_cpu.sh

sbatch run_active_clustering_usps_cpu.sh

sbatch run_active_clustering_pendigits_cpu.sh

sbatch run_active_clustering_bloodmnist_cpu.sh

sbatch run_active_clustering_mnist_cpu.sh

sbatch run_active_clustering_fashion_cpu.sh