# save_strfs.py
# author: Tim Currier; last updated: 2022-05-25
# saves numpy arrays and 2-color mp4 movies of the STRFs for all ROIs in a set
# array filter is 4 sec; movie of the first 2 sec at 1/2 real-time speed

# argumenmts: [1] date (yyyy-mm-dd); [2] series_number; [3] roi_set_name
# implementation: save_strfs.py 2022-03-17 1 roi_set_post

# import relevant packages
from visanalysis.analysis import imaging_data, shared_analysis
from tac_util import tac_h5_tools
import numpy as np
import os
import sys
import cv2
import warnings
from pathlib import Path


# disable runtime and deprecation warnings - dangerous! turn this off while working on the function
warnings.filterwarnings("ignore")

# define recording series to analyze
experiment_file_directory = '/Volumes/TBD/Bruker/StimData'
experiment_file_name = sys.argv[1]
series_number = int(sys.argv[2])
roi_set_name = sys.argv[3]

# join path to proper format for ImagingDataObject()
file_path = os.path.join(experiment_file_directory, experiment_file_name + '.hdf5')

# create save directory
save_directory = '/Volumes/TBD/Bruker/STRFs/' + experiment_file_name + '/'
Path(save_directory).mkdir(exist_ok=True)

# load frame timing offsets for the series; these will be added onto roi_data['time_vector'] to correct for timing dif
frame_offsets = tac_h5_tools.get_vol_frame_offsets(file_path, series_number)

# create ImagingDataObject (wants a path to an hdf5 file and a series number from that file)
ID = imaging_data.ImagingDataObject(file_path,
                                    series_number,
                                    quiet=True)

# get ROI timecourses and stimulus parameters
roi_data = ID.getRoiResponses(roi_set_name);
epoch_parameters = ID.getEpochParameters();
run_parameters = ID.getRunParameters();

# pull run parameters (same across all trials)
update_rate = run_parameters['update_rate']
rand_min = run_parameters['rand_min']
rand_max = run_parameters['rand_max']

# calculate size of noise grid, in units of patches (H,W)
output_shape = (int(np.floor(run_parameters['grid_height']/run_parameters['patch_size'])), int(np.floor(run_parameters['grid_width']/run_parameters['patch_size'])));

# calculate number of time points needed
n_frames = update_rate*(run_parameters['stim_time']);

# initialize array that will contain stimuli for all trials
all_stims = np.zeros(output_shape+(int(n_frames),int(run_parameters['num_epochs'])))

# populate all-trial stimulus array
for trial_num in range(1, roi_data['epoch_response'].shape[1]+1):
# for trial_num in range(1, int(run_parameters['num_epochs']+1)):
    # pull start_seed for trial
    start_seed = epoch_parameters[(trial_num-1)]['start_seed']
    # initialize stimulus frames variable with full idle color
    stim = np.full(output_shape+(int(n_frames),),run_parameters['idle_color'])
    # populate stim array (H,W,T) specifically during "stim time"
    for stim_ind in range(0, stim.shape[2]):
        # find time in sec at stim_ind
        t = (stim_ind+3)/update_rate;
        # define seed at each timepoint
        seed = int(round(start_seed + t*update_rate))
        np.random.seed(seed)
        # find random values for the current seed and write to pre-initialized stim array
        if run_parameters['rgb_texture']: # this variable tracks if a UV stim is being played
            # if this is a UV series, need to populate full [uv,g,b] stim data and subsample only the UV portion
            rand_values = np.random.choice([rand_min, (rand_min + rand_max)/2, rand_max], (output_shape+(3,)));
            stim[:,:,stim_ind] = rand_values[:,:,0];
        else:
            rand_values = np.random.choice([rand_min, (rand_min + rand_max)/2, rand_max], output_shape);
            stim[:,:,stim_ind] = rand_values;
    # save trial stimulus to all_stims(Height, Width, Time, Trial)
    all_stims[:,:,:,(trial_num-1)] = stim;

# define filter length in seconds, convert to samples
filter_length = 3;
filter_len = (filter_length*run_parameters['update_rate']).astype('int')

# pull timing of all imaging frames, along with the start of each trial
full_frame_time_vector = ID.getResponseTiming()['time_vector']
trial_starts = ID.getStimulusTiming(command_frame_rate=60)['stimulus_start_times']

# define the stim onset-relative stimulus timing vector
stim_t = np.arange(run_parameters['pre_time'],run_parameters['pre_time']+run_parameters['stim_time'],1/run_parameters['update_rate'])

# define the mean stimulus for each trial
stim_means = np.mean(all_stims,axis=2)

# iterate over ROIs to save STRFs and movies
print('Calculating STRFs...')
for roi_id in range(0, roi_data['epoch_response'].shape[0]):

    # initialize an array to hold each trial's response-weighted stimulus history
    trial_strfs = np.zeros((output_shape + (filter_len,roi_data['epoch_response'].shape[1])))

    # populate trial_strfs array with response-weighted stimulus histories
    for trial_num in range(0, roi_data['epoch_response'].shape[1]):
        # define mean-subtracted response for this trial
        current_resp = roi_data['epoch_response'][roi_id,trial_num] - np.mean(roi_data['epoch_response'][roi_id,trial_num])
        # use the only non-empty z-slice of roi_mask to define the slice timing offset index to use
        slice_offset = frame_offsets[np.where(np.any(roi_data['roi_mask'][roi_id], axis=(0,1)))[0][0]]
        # find the index of the first imaging frame after each trial_starts time
        trial_start_ind = np.where(np.diff(np.where(full_frame_time_vector<trial_starts[trial_num],0,1)))[0][0]+1
        trial_offset = full_frame_time_vector[trial_start_ind]-trial_starts[trial_num]
        # correct imaging times for this roi, then accept only frames where the full history contains stimulus
        corrected_img_t = roi_data['time_vector']+trial_offset+slice_offset
        cropped_img_t = corrected_img_t[np.logical_and(corrected_img_t > (run_parameters['pre_time']+filter_length), corrected_img_t < (run_parameters['pre_time']+run_parameters['stim_time']))]
        # crop response to same frames
        cropped_resp = current_resp[np.logical_and(corrected_img_t > (run_parameters['pre_time']+filter_length), corrected_img_t < (run_parameters['pre_time']+run_parameters['stim_time']))]
        # initialize array to hold stimulus histories for each time-cropped imaging frame (dims = h,w,t,frame)
        stim_history = np.zeros((output_shape + (filter_len,len(cropped_img_t))))
        # for each time-cropped imaging frame, pull the last filter_len stim flips
        for frame in range(0,len(cropped_img_t)):
            # find last stim index with time less than the current frame time
            last_stim_flip = np.max(np.where(stim_t < cropped_img_t[frame])[0])
            # take last filter_len stim flips for this imaging frame (inclusive)
            stim_history[:,:,:,frame] = all_stims[:,:,(last_stim_flip-filter_len+1):(last_stim_flip+1),trial_num]
            # subtract trial mean stimulus from each timepoint in the filter
            for t in range(0,filter_len):
                stim_history[:,:,t,frame] = stim_history[:,:,t,frame] - stim_means[:,:,trial_num]
        # calculate the response-weighted stimulus history
        trial_strfs[:,:,:,trial_num] = np.dot(stim_history,cropped_resp)

    # take the mean strf across trials
    roi_mean_strf = np.mean(trial_strfs, axis=3)

    # z-score using the first second of the filter
    roi_mean_strf_z = np.zeros(roi_mean_strf.shape)
    STRF_std = np.std(trial_strfs[:,:,0:run_parameters['update_rate'].astype('int'),:],(2,3))
    # divide by std to generate z-scored STRF
    for frame in range(0,roi_mean_strf.shape[2]):
        roi_mean_strf_z[:,:,frame] = roi_mean_strf[:,:,frame]/STRF_std

    ### WRITE MOVIE ###
    # oversample z-scored STRF in x and y so video looks better
    big_strf=roi_mean_strf_z.repeat(20,axis=0)
    bigger_strf=big_strf.repeat(20,axis=1)
    # oversample z-scored STRF in t so framerate can be reduced
    biggest_strf=bigger_strf.repeat(2,axis=2)
    # split into positive and negative patches, convert to semi-exponential colormap
    pos_strf=np.power(np.where(biggest_strf>0,biggest_strf,0),1.5)
    neg_strf=np.power(np.where(biggest_strf<0,biggest_strf*-1,0),1.5)
    # convert z-scored STRF to -1 to 1 scale (units are standard deviations)
    low_lim = -4
    high_lim = 4
    p_new_strf = ((pos_strf - low_lim) * (2/(high_lim - low_lim))) - 1
    n_new_strf = ((neg_strf - low_lim) * (2/(high_lim - low_lim))) - 1
    p_new_strf=np.where(p_new_strf>1,1,p_new_strf)
    n_new_strf=np.where(n_new_strf>1,1,n_new_strf)
    # make empty rgb array and populate with positive or negative values
    rgb_strf=np.zeros((p_new_strf.shape+(3,)))
    rgb_strf[:,:,:,0]=1-(p_new_strf*1)-(n_new_strf*.3)
    rgb_strf[:,:,:,2]=1-(p_new_strf*1)-(n_new_strf*0)
    rgb_strf[:,:,:,1]=1-(p_new_strf*.3)-(n_new_strf*1)
    rgb_strf=np.where(rgb_strf>1,1,rgb_strf)
    # scale rgb_strf to 0-255
    rgb_strf = (rgb_strf*255).astype('uint8')
    # save multicolor video
    fps = 10
    video = cv2.VideoWriter('/Volumes/TBD/Bruker/STRFs/' + experiment_file_name + '/' + experiment_file_name + '-' + roi_set_name + '_' + str(roi_id) + '_movie.mp4', cv2.VideoWriter_fourcc(*'mp4v'), float(fps), (p_new_strf.shape[1],p_new_strf.shape[0]))
    for frame_count in range(int(2*p_new_strf.shape[2]/3),int(p_new_strf.shape[2])):
        #this currently only plots the last second (2/3 : 3/3) of an un-reveresed filter
        img = rgb_strf[:,:,frame_count,:]
        video.write(img)
    video.release()
print('Done.')
