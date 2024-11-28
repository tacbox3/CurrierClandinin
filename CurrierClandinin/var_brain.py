# var_brain.py
# author: TAC
# last update: 20230328
#
# Saves a mean-normalized, thresholded "variance brain" to a specified directory.
# Designed to run immediately after brainsss motion correction (assumes brainsss std output file structure), modulo recover_isotropy and subtract_stimulus
#
# required arguments: [1] series_name; [2] threshold; [3] bin_bool; [4] bgs_bool
# usage: python3 var_brain.py TSeries-20220317-003 650 1 0


# import relevant packages
import numpy as np
import nibabel as nib
import os
import sys

# define recording series to load
base_file_directory = '/oak/stanford/groups/trc/data/Tim/ImagingData/processed/'
experiment = sys.argv[1]
experiment_date = experiment[8:16]
series_number = experiment[17:20]

# build save path
save_path = os.path.join(base_file_directory,experiment_date,'moco',experiment_date+'_'+series_number+'_thresh_brain')

# build path and load .nii as numpy array (width, height, depth, time) based on boolean arguments
bin_bool = sys.argv[3]
bgs_bool = sys.argv[4]
if bin_bool == '1':
    if bgs_bool == '1':
        chan1_path=os.path.join(base_file_directory,experiment_date,'moco','TSeries-'+experiment_date+'-'+ series_number+'_channel_1_binned_bgs_moco.nii')
        chan2_path=os.path.join(base_file_directory,experiment_date,'moco','TSeries-'+experiment_date+'-'+ series_number+'_channel_2_binned_bgs_moco.nii')
    else:
        chan1_path=os.path.join(base_file_directory,experiment_date,'moco','TSeries-'+experiment_date+'-'+ series_number+'_channel_1_binned_moco.nii')
        chan2_path=os.path.join(base_file_directory,experiment_date,'moco','TSeries-'+experiment_date+'-'+ series_number+'_channel_2_binned_moco.nii')
else:
    if bgs_bool == '1':
        chan1_path=os.path.join(base_file_directory,experiment_date,'moco','TSeries-'+experiment_date+'-'+ series_number+'_channel_1_bgs_moco.nii')
        chan2_path=os.path.join(base_file_directory,experiment_date,'moco','TSeries-'+experiment_date+'-'+ series_number+'_channel_2_bgs_moco.nii')
    else:
        chan1_path=os.path.join(base_file_directory,experiment_date,'moco','TSeries-'+experiment_date+'-'+ series_number+'_channel_1_moco.nii')
        chan2_path=os.path.join(base_file_directory,experiment_date,'moco','TSeries-'+experiment_date+'-'+ series_number+'_channel_2_moco.nii')
if os.path.exists(chan2_path):
    print('Two channels detected, processing functional channel (2).')
    nib_brain=np.asanyarray(nib.load(chan2_path).dataobj)
else:
    print('Single channel scan detected.')
    nib_brain=np.asanyarray(nib.load(chan1_path).dataobj)
print('Calculating variance brain...')

# define variance, mean, and fano brains
var_brain=np.swapaxes(np.var(nib_brain,3),0,1);
mean_brain=np.swapaxes(np.mean(nib_brain,3),0,1);
fano_brain=var_brain/mean_brain;
print('Variance brain calculated. Saving...')

# define and apply threshold
threshold=int(sys.argv[2])
thresh_brain=np.where(fano_brain>threshold,fano_brain,1);

# save array as nifti
nib.Nifti1Image(np.swapaxes(thresh_brain,0,1), np.eye(4)).to_filename(save_path)
print('Variance brain saved.')
