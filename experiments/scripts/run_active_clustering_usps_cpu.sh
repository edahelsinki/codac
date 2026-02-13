#!/bin/bash
#SBATCH --job-name=actclust-usps
#SBATCH --time=00:30:00
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3G
#SBATCH --array=0-9
#SBATCH --output=logs/%x_%A_%a.txt

# This is a sbatch script to be executed on a cluster using slurm
# `sbatch [filename].sh`

BIN="/home/patron/bin/conda/envs/clust/bin/python"
cd ..
$BIN "eval_active_clustering.py" -o $RESULTS -n -d usps -c $CONFIG --job-index $SLURM_ARRAY_TASK_ID -q 800