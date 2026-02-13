#!/bin/bash

# export enviroment variables for slurm scripts

# results directory in experiments/results where the results are written
export RESULTS="heatmap_part2"
# configuration for models
export CONFIG="configs/ablation_loss_weights_heatmap_part2.yaml"

sbatch run_active_clustering_fashion_heatmap.sh