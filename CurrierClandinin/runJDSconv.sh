#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=tac_ds_sim
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=232GB
#SBATCH --partition=trc
#SBATCH --output=/home/users/currier/job_outputs/%x.%j.out
#SBATCH --open-mode=append

# Run DS_simulate.py. For all neurons in the specified data file, simulate directional motion responses via linear convolution. Data file should be stored in /oak/stanford/groups/trc/data/Tim/ImagingData/processed/
# Params (passed directly to DS_simulate): (1) spatial period, (2) temporal period, (3) number of orientations
# USAGE: sbatch runDSconv.sh 40 1 8

date
ml python/3.6.1
# apparently SciPy and NumPy are included in the python3 module, so they don't need to be loaded separately

spatial_period=$1
temporal_period=$2
num_angles=$3

python3 -u /home/users/currier/tac_analysis/scripts/DS_simulate_jointSTRF.py $spatial_period $temporal_period $num_angles
