#!/bin/bash

# export enviroment variables for slurm scripts

# results directory in experiments/results where the results are written
export RESULTS="train_test"
# configuration for models
export CONFIG="configs/active_clustering_codac.yaml"

# start the jobs for each dataset

sbatch run_active_clustering_segmentation_test.sh

sbatch run_active_clustering_webkb_test.sh

sbatch run_active_clustering_handwritten_test.sh

sbatch run_active_clustering_optdigits_test.sh

sbatch run_active_clustering_waveform_test.sh

sbatch run_active_clustering_har_test.sh

sbatch run_active_clustering_usps_test.sh

sbatch run_active_clustering_pendigits_test.sh

sbatch run_active_clustering_bloodmnist_test.sh

sbatch run_active_clustering_reuters_test.sh

sbatch run_active_clustering_mnist_test.sh

sbatch run_active_clustering_fashion_test.sh