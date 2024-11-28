#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=moco_tac_brainsss
#SBATCH --time=1-12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=trc
#SBATCH --output=/home/users/currier/job_outputs/%x.%j.out
#SBATCH --open-mode=append

# For a single TSeries, this batch file will bin data that has been oversampled in X, run brainss motion correction, then calculate a variance brain. Binning is only performed if the third argument is '1'.
# Params: (1) base directory, (2) series base name (no suffixes), (3) optional - type_of_transform, (4) optional - output_format, (5) optional - meanbrain_n_frames
# USAGE:
#    sbatch moco_1chan.sh /oak/stanford/groups/trc/data/Tim/ImagingData/processed/YYYYMMDD/ TSeries-YYYYMMDD-00N

date
ml python/3.6 py-ants/0.3.2_py36

directory=$1
series_base=$2

python3 -u /home/users/currier/tac_analysis/scripts/recover_isotropy.py $series_base
brain_master="${series_base}_channel_1_binned.nii"

echo $directory
echo $brain_master

# Optional params
type_of_transform=${3:-"Rigid"}
output_format=${4:-"nii"}
meanbrain_n_frames="${5:-"100"}"

echo $type_of_transform
echo $output_format
echo $meanbrain_n_frames

moco_directory="${directory}/moco/"
args="{\"directory\":\"$directory\",\"brain_master\":\"$brain_master\","\
"\"type_of_transform\":\"$type_of_transform\",\"output_format\":\"$output_format\",\"meanbrain_n_frames\":\"$meanbrain_n_frames\"}"

python3 -u /home/users/currier/brainsss/scripts/motion_correction.py $args
python3 -u /home/users/currier/tac_analysis/scripts/var_brain.py $series_base 750 1 0
