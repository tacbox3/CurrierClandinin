# This resource-intensive script simulates DS measurements by convolving the STRFs of all neurons in a given dataset with a moving sinusoidal grating. The .py file containing STRF data must exist in the same folder as this script. Indeded to be called via batch script on Sherlock.
# Takes input arguments specifying the spatial and temporal periods of the simulated stimulus, as well as the number of orientations to test (must be an even divisor of 360)
# Required arguments: [1] spatial period (deg) [2] temporal period (deg) [3] number of orientations
# Implementation: DS_simulate.py 40 1 36

import numpy as np
import os
import sys
from scipy import ndimage
from scipy import signal
from pathlib import Path

# load STRF data
base_file_directory = '/oak/stanford/groups/trc/data/Tim/ImagingData/processed/'
data_path = os.path.join(base_file_directory,'all_smoothed_strfs.npy')
all_centered_STRFs = np.load(data_path)
print('Loaded STRF data. Constructing artificial stimulus...')

# stimulus paramters
spatial_period = int(sys.argv[1]) #deg (equivalent to indices)
temporal_period = float(sys.argv[2]) #sec
screen_size = 80 #deg
num_angles = int(sys.argv[3]) #number of orientations to simulate - must be an even divisor of 360
stim_duration = 10 #sec
pre_time = 5 #sec
tail_time = 5 #sec
dt = 0.05 #sec

# container arrays for moving stimulus (dims = x,y,t) and oriented moving stimulus (x,y,t,ang). s is double sized in x and y relative to s_orient to account for clipping when using ndimage.rotate()
s = np.zeros((screen_size*2,screen_size*2,int(stim_duration/dt)))
s_orient = np.zeros((screen_size,screen_size,int(stim_duration/dt),int(360/num_angles)))

# build stimulus
spatial_period = spatial_period/2 #corrects for screen subsetting during rotation
n_spatial_cyc = screen_size/spatial_period
x = np.arange(-n_spatial_cyc*np.pi,n_spatial_cyc*np.pi,2*n_spatial_cyc*np.pi/(screen_size*2)).astype('float32')
# create initial screen as sinusoid over interval (0,1)
s0 = np.tile((np.cos(x)+1)/2,(screen_size*2,1)).astype('float32') # index in x and s0 is 1/2 deg
s[:,:,0] = s0
# fill in timepoints by shifting the array contents and appropriate number of indices
for t in range(1,int(stim_duration/dt)):
    # in 1 sec, need to move the sinusoid by spatial_period/temporal_period deg
    # in dt sec, need to move it by dt*spatial_period/temporal_period deg
    s[:,:,t] = np.roll(s[:,:,t-1],int(2*dt*spatial_period/temporal_period), axis=1)
print('Timeseries complete. Rotating...')
del s0
del x

# rotate full timecourse in plane defined by first two dimensions; do this num_angles times
for ang_ind in range(0,num_angles):
    # define rotation angle
    rot_ang = int((360/num_angles)*ang_ind)
    # grab the middle screen_size indices of the non-reshaped rotated array - this puts the total coverage back to screen_size deg with a density of 2 indices per deg
    s_orient[:,:,:,ang_ind] = ndimage.rotate(s,rot_ang,reshape=False) [int(screen_size*1/2):int(screen_size*3/2),int(screen_size*1/2):int(screen_size*3/2),:]
print('Rotation complete. Appending gray screen to pre- and tail- periods...')
del s

# append gray screen (0.5) to beginning and end of rotated stimulus array
pre_pad = 0.5*np.ones((screen_size,screen_size,int(pre_time/dt),int(360/num_angles)))
post_pad = 0.5*np.ones((screen_size,screen_size,int(tail_time/dt),int(360/num_angles)))
center_s_orient = np.append(np.append(pre_pad, s_orient, axis=2), post_pad, axis=2)
print('Artificial stimulus constructed. Performing convolutions...')
del s_orient
del pre_pad
del post_pad

# container for all trimmed and baseline-subtracted convolution responses (dims = time, orientation, cell)
conv_resp = np.full((center_s_orient.shape[2]-int(6/dt), num_angles, all_centered_STRFs.shape[4]), np.nan, dtype='float')

# perform convolution and save the sum over spatial dimensions of STRF for each orientation (output is a timecourse of the total response for each motion direction)
for cell in range(0,all_centered_STRFs.shape[4]):
    if np.remainder(cell,5) == 0:
        print('Current index = ' + str(cell))
    try:
        # only run on cells with both colors recorded
        if not np.any(np.isnan(all_centered_STRFs[:,:,:,:,cell])):
            # take mean STRF over colors as input... this should reduce noise
            current_strf = np.nanmean(all_centered_STRFs[:,:,:,:,cell], axis=3)
            # spatial upsample to match density of stimulus, then grab central half of STRF to match size of stimulus
            upsampled_strf = np.repeat(np.repeat(current_strf,int(2*center_s_orient.shape[0]/current_strf.shape[0]),axis=0),int(2*center_s_orient.shape[1]/current_strf.shape[1]),axis=1)
            center_strf = upsampled_strf[int(center_s_orient.shape[0]/2):int(center_s_orient.shape[0]*3/2),int(center_s_orient.shape[1]/2):int(center_s_orient.shape[1]*3/2),:]
            # iterate over orientations
            for ori in range(0,num_angles):
                # sum convolution products over spatial dimensions (need to flip STRF to align time vectors)
                conv_prod = np.sum(signal.convolve(center_s_orient[:,:,:,ori], np.flip(center_strf,axis=2), mode='same', method='auto'),axis=(0,1))
                # trim off the gray screen edge effects and baseline subtract the convolution product
                trim = conv_prod[int(3/dt):int((pre_time+stim_duration+tail_time-3)/dt)]
                conv_resp[:,ori,cell] = trim - np.mean(trim[:5])
    except:
        print('Error raised while handling cell ' + str(cell) + '.')
        break
    if cell == all_centered_STRFs.shape[4]-1:
        print('DS simulations complete. Saving responses...')

# save output variable to oak directory containing the data file
save_path = os.path.join(base_file_directory, 'jointSTRF_DS_simulations_' + sys.argv[1] + '_' + sys.argv[2] + '_' + sys.argv[3] + '_flip_smoothed.npy')
np.save(save_path, conv_resp)
print('Data saved to ' + save_path)
