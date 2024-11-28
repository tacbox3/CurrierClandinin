#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=tac_proc_imports
#SBATCH --time=99:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=trc
#SBATCH --output=/home/users/currier/job_outputs/%x.%j.out
#SBATCH --open-mode=append

# a simple call to proc_imports for a single folder, given as an argument
# USAGE: sbatch import.sh YYYYMMDD

date
ml python/3.6.1

experiment_date=$1

python3 -u /home/users/currier/tac_analysis/scripts/proc_imports.py $experiment_date
