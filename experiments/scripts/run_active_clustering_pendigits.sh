#!/bin/bash
#SBATCH --job-name=actclust-pendigits
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-gpu=8G
#SBATCH --array=0-9
#SBATCH --output=logs/%x_%A_%a.txt

# This is a sbatch script to be executed on a cluster using slurm
# `sbatch [filename].sh`

BIN="/home/patron/bin/conda/envs/clust/bin/python"
cd ..
$BIN "eval_active_clustering.py" -o $RESULTS -n -d pendigits -c $CONFIG --job-index $SLURM_ARRAY_TASK_ID -q 800