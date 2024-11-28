# recover_isotropy.py
# author: Tim Currier; last updated: 2022-10-25
# Bins pixels into batches of two. Designed for brains that have been oversampled in X to facilitate denoising.
# Best run immediately before brainsss motion correction.

# required arguments: [1] series_name
# usage: python3 recover_isotropy.py TSeries-20220317-003

# import relevant packages
import numpy as np
import nibabel as nib
import os
import sys

# define recording series to load
base_file_directory = '/oak/stanford/groups/trc/data/Tim/ImagingData/processed/'
experiment = sys.argv[1]
experiment_date = experiment[8:16]
date_path = os.path.join(base_file_directory,experiment_date)
file_path = os.path.join(date_path,experiment+'_channel_1.nii')
save_path = os.path.join(date_path,experiment+'_channel_1_binned.nii')

# bin the brain into n/2 X-axis bins, where n is the number of pixels along the X axis
brain = np.asarray(nib.load(file_path).get_data(), dtype='int16')
print('Original brain dims:',brain.shape)
trunc_brain = brain[0:int(np.floor(brain.shape[0]/2)*2), :, :, :]
brain_binned = trunc_brain[0::2, :, :, :] + trunc_brain[1::2, :, :, :]
print('New dims:',brain_binned.shape)

# save binned brain
nib.Nifti1Image(brain_binned, np.eye(4)).to_filename(save_path)
print('Binned brain saved.')

# check for channel_2.nii and perform same operation if it exists
chan2_path = os.path.join(date_path,experiment+'_channel_2.nii')
if os.path.exists(chan2_path):
    print('Channel 2 exists! Binning...')
    chan2_save_path = os.path.join(date_path,experiment+'_channel_2_binned.nii')
    brain2 = np.asarray(nib.load(chan2_path).get_data(), dtype='int16')
    print('Original channel 2 dims:',brain2.shape)
    trunc_brain2 = brain2[0:int(np.floor(brain2.shape[0]/2)*2), :, :, :]
    brain2_binned = trunc_brain2[0::2, :, :, :] + trunc_brain2[1::2, :, :, :]
    print('New dims:',brain2_binned.shape)
    nib.Nifti1Image(brain2_binned, np.eye(4)).to_filename(chan2_save_path)
    print('Binned channel 2 saved.')
else:
    print('No second channel found.')
