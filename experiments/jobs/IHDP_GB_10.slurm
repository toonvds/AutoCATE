#!/usr/bin/env bash

#SBATCH --nodes="1"
#SBATCH --ntasks="16"
#SBATCH --cluster="genius"
#SBATCH --mem="5G"
#SBATCH --time="01:00:00"
#SBATCH --account="lp_stat_cluster"
#SBATCH --mail-type="END,FAIL"
#SBATCH --mail-user="toon.vanderschueren@kuleuven.be"
#SBATCH --output="/vsc-hard-mounts/leuven-data/341/vsc34195/AutoCATE/experiments/results/%x.o%A"
#SBATCH --error="/vsc-hard-mounts/leuven-data/341/vsc34195/AutoCATE/experiments/results/%x.e%A"

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib

source activate AutoCATE
cd $VSC_DATA/AutoCATE/

python -m experiments.experiment_main --dataset=IHDP --experiment_iter=100 --n_folds=1 --n_trials=10 --n_eval_versions=1 --n_eval_ensemble=1 --n_eval_trials=10 --no-AutoCATE --no-random_forest --gradient_boosting --no-linear_regression
