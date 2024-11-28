from visanalysis.analysis import imaging_data, shared_analysis
from tac_util import tac_h5_tools
import numpy as np
import os
import sys
import xlrd
import warnings
from pathlib import Path
from datetime import datetime
from scipy.optimize import curve_fit

# defines the curve(s) to fit nonlinearities to; sigmoid will be tried first, line is applied if a fit to the sigmoid can't be found
def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return (y)
def line(x, k, b):
    y = k*x + b
    return (y)


book = xlrd.open_workbook('/Volumes/TBD/Bruker/ME0708_full_log_snap.xls')
sheet = book.sheet_by_name('Sheet1')
data = [[sheet.cell_value(r, c) for c in range(sheet.ncols)] for r in range(sheet.nrows)]
full_log = np.asarray(data)

# basic info for recalculating STRFs
experiment_file_directory = '/Volumes/TBD/Bruker/StimData'
tof = 2 #temporal oversample factor for binning stimulus histories
grid_size = (16,16)
filter_length = 3 #sec - this was run with 4 initially, but I want to minimize signal-less data
stim_update_rate = 20 #Hz
nbins = 20 #number of bins to calculate mean nonlinearity
filter_len = filter_length * stim_update_rate * tof
flick_pre = 2 #sec
flick_tail = 2 #sec
flick_stim = 10 #sec
flick_duration = flick_pre + flick_tail + flick_stim #sec
flick_conditions = 4
flick_resample_rate = 5 #Hz (data will be interpolated to this freq to correct variable framerate)


# reduce full log to rows containing H5-tagged ROIs
H5_bool = full_log[:,0] == 'o'
scraped_log = full_log[H5_bool,:]

# repopulate date strings from excel sequential day format
for roi_ind in range(0, np.shape(scraped_log)[0]):
    excel_date = int(float(scraped_log[roi_ind,2]))
    dt = datetime.fromordinal(datetime(1900, 1, 1).toordinal() + excel_date - 2)
    scraped_log[roi_ind,2] = str(dt)[0:10]

# generate resample time vector for flicker responses
flick_resamp_t = np.arange(0, flick_duration-0.8, 1/flick_resample_rate)

# populate NaN arrays for blue, uv, and flicker responses, as well as filter nonlinearities
all_blue_resp = np.full(grid_size+(int(filter_len/tof),np.shape(scraped_log)[0]), np.NaN, dtype='float')
all_blue_resp_dff = np.full(grid_size+(int(filter_len/tof),np.shape(scraped_log)[0]), np.NaN, dtype='float')
all_blue_mean_nl = np.full((nbins,2,2,(np.shape(scraped_log)[0])), np.NaN, dtype='float')
all_blue_nl_fits = np.full((4,2,(np.shape(scraped_log)[0])), np.NaN, dtype='float')
all_uv_resp = np.full(grid_size+(int(filter_len/tof),np.shape(scraped_log)[0]), np.NaN, dtype='float')
all_uv_resp_dff = np.full(grid_size+(int(filter_len/tof),np.shape(scraped_log)[0]), np.NaN, dtype='float')
all_uv_mean_nl = np.full((nbins,2,2,(np.shape(scraped_log)[0])), np.NaN, dtype='float')
all_uv_nl_fits = np.full((4,2,(np.shape(scraped_log)[0])), np.NaN, dtype='float')
all_flick_resp = np.full((flick_conditions,len(flick_resamp_t),np.shape(scraped_log)[0]), np.NaN, dtype='float')

#%% load array with centroid coordinates (units = deg) and convert shifts to actual patch indices
cent_shifts = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all_shifts.npy')[0:2,:]
cent_inds = (-1*cent_shifts) + 40

# compress to patch number units
cent_inds = np.round(cent_inds/5)

# %% for each line in the scraped log, reconstruct the blue STRF
# this takes about 8 sec per ROI to run (for ~350 ROIS, ~45 min)
# for roi_ind in range(6, 7):
for roi_ind in range(0, np.shape(scraped_log)[0]):
    # check if the curent ROI exists
    if scraped_log[roi_ind,4] != '':
        # first, pull important details from spreadsheet
        file_path = os.path.join(experiment_file_directory, scraped_log[roi_ind,2] + '.hdf5')
        series_number = int(float(scraped_log[roi_ind,3]))
        roi_set_name = scraped_log[roi_ind,4]
        roi_num = int(float(scraped_log[roi_ind,5]))
        # create ImagingDataObject for the series
        ID = imaging_data.ImagingDataObject(file_path, series_number, quiet=True)
        # determine stimulus frame rate for fly
        if np.isnan(ID.getStimulusTiming(command_frame_rate=60)['frame_rate']):
            stim_sr = 120
        else:
            stim_sr = 60
        # get ROI timecourses and stimulus parameters
        roi_data = ID.getRoiResponses(roi_set_name);
        epoch_parameters = ID.getEpochParameters();
        run_parameters = ID.getRunParameters();
        # use the only non-empty z-slice of roi_mask to define the slice timing offset index to use
        slice_offset = tac_h5_tools.get_vol_frame_offsets(file_path, series_number)[np.where(np.any(roi_data['roi_mask'][roi_num], axis=(0,1)))[0][0]]
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
        for trial_num in range(1, int(run_parameters['num_epochs']+1)):
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
                # THIS IS ONLY FOR BLUE!
                rand_values = np.random.choice([rand_min, (rand_min + rand_max)/2, rand_max], output_shape);
                stim[:,:,stim_ind] = rand_values;
            # save trial stimulus to all_stims(Height, Width, Time, Trial)
            all_stims[:,:,:,(trial_num-1)] = stim;

        # oversample stimulus by replicating it tof times
        to_stims = np.repeat(all_stims, tof, axis=2)

        # pull timing information for all imaging and stimulus frames
        full_frame_time_vector = ID.getResponseTiming()['time_vector']
        trial_starts = ID.getStimulusTiming(command_frame_rate=stim_sr)['stimulus_start_times']
        trial_ends = ID.getStimulusTiming(command_frame_rate=stim_sr)['stimulus_end_times']
        all_frames = ID.getStimulusTiming(command_frame_rate=stim_sr)['frame_times']

        # initialize an array to hold each trial's response-weighted stimulus
        trial_strfs = np.zeros((output_shape + (filter_len,roi_data['epoch_response'].shape[1])))

        # define the mean stimulus for each trial
        stim_means = np.mean(to_stims,axis=2)

        # populate trial_strfs array with response-weighted stimulus histories
        for trial_num in range(0, roi_data['epoch_response'].shape[1]):
            # define mean-subtracted response for this trial
            current_resp = roi_data['epoch_response'][roi_num,trial_num] - np.mean(roi_data['epoch_response'][roi_num,trial_num])

            # grab imaging frame times for current trial
            img_t = full_frame_time_vector[np.logical_and(full_frame_time_vector >= trial_starts[trial_num]-run_parameters['pre_time'], full_frame_time_vector <= trial_ends[trial_num]-run_parameters['pre_time'])]

            # trim response to match length of imaging timepoints vector
            current_resp = current_resp[:len(img_t)]

            # grab stim frame times for current trial, retain every stim flip index, which is a function of stim_sr. then, resample according to tof
            stim_t = all_frames[np.logical_and(all_frames >= trial_starts[trial_num], all_frames <= trial_ends[trial_num])][0::int(stim_sr/20)]
            stim_t = np.interp(np.arange(0,len(stim_t),1/tof), np.arange(0,len(stim_t),1), stim_t)

            # convert img and stim times to trial-relative units
            trial_stim_t = stim_t-stim_t[0]
            trial_img_t = img_t-stim_t[0]+slice_offset

            # accept only frames where the full history contains stimulus
            cropped_img_t = trial_img_t[np.logical_and(trial_img_t > (filter_length), trial_img_t < (run_parameters['stim_time']))]
            # crop response to same frames
            cropped_resp = current_resp[np.logical_and(trial_img_t > (filter_length), trial_img_t < (run_parameters['stim_time']))]

            # initialize array to hold stimulus histories for each time-cropped imaging frame (dims = h,w,t,frame)
            stim_history = np.zeros((output_shape + (filter_len,len(cropped_img_t))))

            # for each time-cropped imaging frame, pull the last filter_len stim flips
            for frame in range(0,len(cropped_img_t)):
                # find last stim index with time less than the current frame time
                last_stim_flip = np.max(np.where(trial_stim_t < cropped_img_t[frame])[0])
                # take last filter_len stim flips for this imaging frame (inclusive)
                stim_history[:,:,:,frame] = to_stims[:,:,(last_stim_flip-filter_len+1):(last_stim_flip+1),trial_num]
                # subtract trial mean stimulus from each timepoint in the filter
                for t in range(0,filter_len):
                    stim_history[:,:,t,frame] = stim_history[:,:,t,frame] - stim_means[:,:,trial_num]

            # calculate the response-weighted stimulus history
            trial_strfs[:,:,:,trial_num] = np.dot(stim_history,cropped_resp)

        # take the mean strf across trials
        roi_mean_strf = np.mean(trial_strfs, axis=3)

        # z-score using the first second of the filter
        roi_mean_strf_z = np.zeros(roi_mean_strf.shape)
        STRF_std = np.std(trial_strfs[:,:,0:run_parameters['update_rate'].astype('int')*tof,:],(2,3))
        # divide by std to generate z-scored STRF (already mean-subtracted)
        for frame in range(0,roi_mean_strf.shape[2]):
            roi_mean_strf_z[:,:,frame] = roi_mean_strf[:,:,frame]/STRF_std

        # collapse back into proper timing density
        roi_mean_strf_z = (roi_mean_strf_z[:,:,0::tof]+roi_mean_strf_z[:,:,1::tof])/tof
        roi_mean_strf = (roi_mean_strf[:,:,0::tof]+roi_mean_strf[:,:,1::tof])/tof

        # add ROI STRF to all_blue_resp
        all_blue_resp[:,:,:,roi_ind] = roi_mean_strf_z
        all_blue_resp_dff[:,:,:,roi_ind] = roi_mean_strf

        # # NONLINEARITY CALCULATION BLOCK
        # # pull all responses and mean-subtract, to compare to linear predictions
        # trial_responses = roi_data['epoch_response'][roi_num,:,:]
        # ms_measured = np.zeros(trial_responses.shape)
        # for n in range(0,trial_responses.shape[0]):
        #     ms_measured[n,:] = trial_responses[n,:]-np.mean(trial_responses[n,:])
        #
        # # create pre/tail zero vector for padding each trial's stimulus
        # pt_pad = np.full((int(run_parameters['pre_time']*run_parameters['update_rate']),1),0.5)
        #
        # # define timepoints of padded convolution (used to resample the prediction at imaging rate)
        # # the weird start and stop time account for the 3-frame lag and indexing shift (-1)
        # conv_t = np.arange(2/run_parameters['update_rate'],49+1/run_parameters['update_rate'],1/run_parameters['update_rate'])
        #
        # # initialize array to hold convolution products (h,w,trial,time), which will have non-valid overhangs trimmed out
        # lin_preds = np.zeros((all_stims.shape[0:2] + all_stims.shape[3:] + (all_stims.shape[2] + 2*len(pt_pad) - 1,)))
        # ds_preds = np.zeros((all_stims.shape[0:2] + all_stims.shape[3:] + (len(roi_data['time_vector']),)))
        #
        # # perform patch-wise convolution, stimulus and lin_preds output will have same timing
        # for w in range(0,all_stims.shape[0]):
        #     for h in range(0,all_stims.shape[1]):
        #         for trial in range(0,all_stims.shape[3]):
        #             current_TRF = trial_strfs[h,w,:,trial]
        #             current_stim = np.append(pt_pad,np.append(all_stims[h,w,:,trial],pt_pad))-0.5
        #             lin_preds[h,w,trial,:] = np.convolve(current_stim, current_TRF, mode='full')[int(len(current_TRF)/2):current_stim.shape[0]-1+int(len(current_TRF)/2)]
        #             # evaluate the interpolated prediction at imaging frame times
        #             ds_preds[h,w,trial,:] = np.interp(cfts[trial_num,:],conv_t,lin_preds[h,w,trial,:])
        #
        # # calculate mean prediction by averaging over patches (resulting dims = trials, time), then trim mean gray periods
        # joint_pred = np.mean(ds_preds[:,:,:,:],axis=(0,1))
        #
        # # sort and flatten the measured and predicted arrays using the sorted measured array indices
        # flat_pred = joint_pred.flatten()
        # flat_measured = ms_measured.flatten()
        # sorted_inds = np.argsort(flat_pred, axis=None)
        # sorted_pred = flat_pred[sorted_inds]
        # sorted_measured = flat_measured[sorted_inds]
        #
        # # find the binned averages for predicted and measured responses
        # nbin_width = int(len(sorted_measured)/nbins)
        # pred_binmeans = np.zeros((nbins,1))
        # measured_binmeans = np.zeros((nbins,1))
        # for bin in range(0,nbins):
        #     measured_binmeans[bin] = np.mean(sorted_measured[nbin_width*bin:nbin_width*(bin+1)])
        #     pred_binmeans[bin] = np.mean(sorted_pred[nbin_width*bin:nbin_width*(bin+1)])
        #
        # # try to fit a sigmoid to the data cloud defined by (predicted,measured); if no fit can be found due to the data's shape, fit a line instead
        # try:
        #     xdata = sorted_pred
        #     ydata = sorted_measured
        #     p0 = [np.max(ydata), 0, 1, 0] #initial conditions for gradient descent
        #     popt, pcov = curve_fit(sigmoid, xdata, ydata, p0, method='lm')
        #     all_blue_nl_fits[:,0,roi_ind] = popt
        # except:
        #     xdata = sorted_pred
        #     ydata = sorted_measured
        #     p0 = [1, 0] #initial conditions for gradient descent
        #     popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
        #     all_blue_nl_fits[2:4,0,roi_ind] = popt
        #
        # # repeat these calculations using the 'bounding box' method, subsampling the STRF 5 samples in all directions from the previously calculated centroid (if the edge of the array is reached in any direction, truncate the size of the bounding box)
        # joint_pred = np.mean(ds_preds[np.max([int(cent_inds[0,roi_ind])-4,0]): np.min([int(cent_inds[0,roi_ind])+4,16]),np.max([int(cent_inds[1,roi_ind])-4,0]): np.min([int(cent_inds[1,roi_ind])+4,16]),:,:],axis=(0,1))
        #
        # flat_pred = joint_pred.flatten()
        # flat_measured = ms_measured.flatten()
        # sorted_inds = np.argsort(flat_pred, axis=None)
        # bb_sorted_pred = flat_pred[sorted_inds]
        # bb_sorted_measured = flat_measured[sorted_inds]
        # bb_pred_binmeans = np.zeros((nbins,1))
        # bb_measured_binmeans = np.zeros((nbins,1))
        # for bin in range(0,nbins):
        #     bb_measured_binmeans[bin] = np.mean(bb_sorted_measured[nbin_width*bin:nbin_width*(bin+1)])
        #     bb_pred_binmeans[bin] = np.mean(bb_sorted_pred[nbin_width*bin:nbin_width*(bin+1)])
        #
        # # try to fit a sigmoid to the data cloud defined by (predicted,measured); if no fit can be found due to the data's shape, fit a line instead
        # try:
        #     xdata = bb_sorted_pred
        #     ydata = bb_sorted_measured
        #     p0 = [np.max(ydata), 0, 1, 0] #initial conditions for gradient descent
        #     bb_popt, bb_pcov = curve_fit(sigmoid, xdata, ydata, p0, method='lm')
        #     all_blue_nl_fits[:,1,roi_ind] = bb_popt
        # except:
        #     xdata = bb_sorted_pred
        #     ydata = bb_sorted_measured
        #     p0 = [1, 0] #initial conditions for gradient descent
        #     bb_popt, bb_pcov = curve_fit(line, xdata, ydata, p0, method='lm')
        #     all_blue_nl_fits[2:4,1,roi_ind] = bb_popt
        #
        # # output variables are: mean_nl, a 3D array with 1st dimension 'bin', 2nd dimension holding the predicted and measured mean values, and 3rd dimension 'prediction type'; and sig_params, a 2D array with 1st dimension 'parameter' including L (curve height), x0 (curve midpoint), k (slope) and b (vertical bias) for the sigmoid function, and 2nd dimension 'prediction type'. To recover the nl curves, plot mean_nl[:,1,n] against mean_nl[:,0,n]. To recover the fit curve, redefine the sigmoid function in the top block
        # mean_nl = np.zeros(all_blue_mean_nl.shape[0:3])
        # mean_nl[:,:,0] = np.append(measured_binmeans,pred_binmeans,axis=1)
        # mean_nl[:,:,1] = np.append(bb_measured_binmeans,bb_pred_binmeans,axis=1)
        # all_blue_mean_nl[:,:,:,roi_ind] = mean_nl



# save mean, dc-subtracted, z-scored STRF, and nonlinearity summaries
np.save('/Volumes/TBD/Bruker/STRFs/compiled/all_blue_rois.npy', all_blue_resp)
np.save('/Volumes/TBD/Bruker/STRFs/compiled/all_blue_rois_dff.npy', all_blue_resp_dff)
# np.save('/Volumes/TBD/Bruker/STRFs/compiled/all_blue_mean_NLs.npy', all_blue_mean_nl)
# np.save('/Volumes/TBD/Bruker/STRFs/compiled/all_blue_NL_fits.npy', all_blue_nl_fits)


# %% reconstruct all UV STRFs
# this takes about 3 sec per ROI to run (for ~350 ROIS, ~18 min)
# for roi_ind in range(428, 429):
for roi_ind in range(0, np.shape(scraped_log)[0]):
    # check if the curent ROI exists
    if scraped_log[roi_ind,7] != '':
        # first, pull important details from spreadsheet
        file_path = os.path.join(experiment_file_directory, scraped_log[roi_ind,2] + '.hdf5')
        series_number = int(float(scraped_log[roi_ind,6]))
        roi_set_name = scraped_log[roi_ind,7]
        roi_num = int(float(scraped_log[roi_ind,8]))
        # create ImagingDataObject for the series
        ID = imaging_data.ImagingDataObject(file_path, series_number, quiet=True)
        # determine stimulus frame rate for fly
        if np.isnan(ID.getStimulusTiming(command_frame_rate=60)['frame_rate']):
            stim_sr = 120
        else:
            stim_sr = 60
        # get ROI timecourses and stimulus parameters
        roi_data = ID.getRoiResponses(roi_set_name);
        epoch_parameters = ID.getEpochParameters();
        run_parameters = ID.getRunParameters();
        # use the only non-empty z-slice of roi_mask to define the slice timing offset index to use
        slice_offset = tac_h5_tools.get_vol_frame_offsets(file_path, series_number)[np.where(np.any(roi_data['roi_mask'][roi_num], axis=(0,1)))[0][0]]
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
        for trial_num in range(1, int(run_parameters['num_epochs']+1)):
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
                rand_values = np.random.choice([rand_min, (rand_min + rand_max)/2, rand_max], (output_shape+(3,)));
                stim[:,:,stim_ind] = rand_values[:,:,0];

            # save trial stimulus to all_stims(Height, Width, Time, Trial)
            all_stims[:,:,:,(trial_num-1)] = stim;

        # oversample stimulus by replicating it tof times
        to_stims = np.repeat(all_stims, tof, axis=2)

        # pull timing information for all imaging and stimulus frames
        full_frame_time_vector = ID.getResponseTiming()['time_vector']
        trial_starts = ID.getStimulusTiming(command_frame_rate=stim_sr)['stimulus_start_times']
        trial_ends = ID.getStimulusTiming(command_frame_rate=stim_sr)['stimulus_end_times']
        all_frames = ID.getStimulusTiming(command_frame_rate=stim_sr)['frame_times']

        # initialize an array to hold each trial's response-weighted stimulus
        trial_strfs = np.zeros((output_shape + (filter_len,roi_data['epoch_response'].shape[1])))

        # define the mean stimulus for each trial
        stim_means = np.mean(to_stims,axis=2)

        # populate trial_strfs array with response-weighted stimulus histories
        for trial_num in range(0, roi_data['epoch_response'].shape[1]):
            # define mean-subtracted response for this trial
            current_resp = roi_data['epoch_response'][roi_num,trial_num] - np.mean(roi_data['epoch_response'][roi_num,trial_num])

            # grab imaging frame times for current trial
            img_t = full_frame_time_vector[np.logical_and(full_frame_time_vector >= trial_starts[trial_num]-run_parameters['pre_time'], full_frame_time_vector <= trial_ends[trial_num]-run_parameters['pre_time'])]

            # trim response to match length of imaging timepoints vector
            current_resp = current_resp[:len(img_t)]

            # grab stim frame times for current trial, retain every third index, then resample according to tof
            stim_t = all_frames[np.logical_and(all_frames >= trial_starts[trial_num], all_frames <= trial_ends[trial_num])][0::int(stim_sr/20)]
            stim_t = np.interp(np.arange(0,len(stim_t),1/tof), np.arange(0,len(stim_t),1), stim_t)

            # convert img and stim times to trial-relative units
            trial_stim_t = stim_t-stim_t[0]
            trial_img_t = img_t-stim_t[0]+slice_offset

            # accept only frames where the full history contains stimulus
            cropped_img_t = trial_img_t[np.logical_and(trial_img_t > (filter_length), trial_img_t < (run_parameters['stim_time']))]
            # crop response to same frames
            cropped_resp = current_resp[np.logical_and(trial_img_t > (filter_length), trial_img_t < (run_parameters['stim_time']))]

            # initialize array to hold stimulus histories for each time-cropped imaging frame (dims = h,w,t,frame)
            stim_history = np.zeros((output_shape + (filter_len,len(cropped_img_t))))

            # for each time-cropped imaging frame, pull the last filter_len stim flips
            for frame in range(0,len(cropped_img_t)):
                # find last stim index with time less than the current frame time
                last_stim_flip = np.max(np.where(trial_stim_t < cropped_img_t[frame])[0])
                # take last filter_len stim flips for this imaging frame (inclusive)
                stim_history[:,:,:,frame] = to_stims[:,:,(last_stim_flip-filter_len+1):(last_stim_flip+1),trial_num]
                # subtract trial mean stimulus from each timepoint in the filter
                for t in range(0,filter_len):
                    stim_history[:,:,t,frame] = stim_history[:,:,t,frame] - stim_means[:,:,trial_num]

            # calculate the response-weighted stimulus history
            trial_strfs[:,:,:,trial_num] = np.dot(stim_history,cropped_resp)

        # take the mean strf across trials
        roi_mean_strf = np.mean(trial_strfs, axis=3)

        # z-score using the first second of the filter
        roi_mean_strf_z = np.zeros(roi_mean_strf.shape)
        STRF_std = np.std(trial_strfs[:,:,0:run_parameters['update_rate'].astype('int')*tof,:],(2,3))
        # divide by std to generate z-scored STRF (already mean-subtracted)
        for frame in range(0,roi_mean_strf.shape[2]):
            roi_mean_strf_z[:,:,frame] = roi_mean_strf[:,:,frame]/STRF_std

        # collapse back into proper timing density
        roi_mean_strf_z = (roi_mean_strf_z[:,:,0::tof]+roi_mean_strf_z[:,:,1::tof])/tof
        roi_mean_strf = (roi_mean_strf[:,:,0::tof]+roi_mean_strf[:,:,1::tof])/tof

        # add ROI STRF to all_uv_resp
        all_uv_resp[:,:,:,roi_ind] = roi_mean_strf_z
        all_uv_resp_dff[:,:,:,roi_ind] = roi_mean_strf

        # # NONLINEARITY CALCULATION BLOCK
        # # pull all responses and mean-subtract, to compare to linear predictions
        # trial_responses = roi_data['epoch_response'][roi_num,:,:]
        # ms_measured = np.zeros(trial_responses.shape)
        # for n in range(0,trial_responses.shape[0]):
        #     ms_measured[n,:] = trial_responses[n,:]-np.mean(trial_responses[n,:])
        #
        # # create pre/tail zero vector for padding each trial's stimulus
        # pt_pad = np.full((int(run_parameters['pre_time']*run_parameters['update_rate']),1),0.5)
        #
        # # define timepoints of padded convolution (used to resample the prediction at imaging rate)
        # # the weird start and stop time account for the 3-frame lag and indexing shift (-1)
        # conv_t = np.arange(2/run_parameters['update_rate'],49+1/run_parameters['update_rate'],1/run_parameters['update_rate'])
        #
        # # initialize array to hold convolution products (h,w,trial,time), which will have non-valid overhangs trimmed out
        # lin_preds = np.zeros((all_stims.shape[0:2] + all_stims.shape[3:] + (all_stims.shape[2] + 2*len(pt_pad) - 1,)))
        # ds_preds = np.zeros((all_stims.shape[0:2] + all_stims.shape[3:] + (len(roi_data['time_vector']),)))
        #
        # # perform patch-wise convolution, stimulus and lin_preds output will have same timing
        # for w in range(0,all_stims.shape[0]):
        #     for h in range(0,all_stims.shape[1]):
        #         for trial in range(0,all_stims.shape[3]):
        #             current_TRF = trial_strfs[h,w,:,trial]
        #             current_stim = np.append(pt_pad,np.append(all_stims[h,w,:,trial],pt_pad))-0.5
        #             lin_preds[h,w,trial,:] = np.convolve(current_stim, current_TRF, mode='full')[int(len(current_TRF)/2):current_stim.shape[0]-1+int(len(current_TRF)/2)]
        #             # evaluate the interpolated prediction at imaging frame times
        #             ds_preds[h,w,trial,:] = np.interp(cfts[trial_num,:],conv_t,lin_preds[h,w,trial,:])
        #
        # # calculate mean prediction by averaging over patches (resulting dims = trials, time), then trim mean gray periods
        # joint_pred = np.mean(ds_preds[:,:,:,:],axis=(0,1))
        #
        # # sort and flatten the measured and predicted arrays using the sorted measured array indices
        # flat_pred = joint_pred.flatten()
        # flat_measured = ms_measured.flatten()
        # sorted_inds = np.argsort(flat_pred, axis=None)
        # sorted_pred = flat_pred[sorted_inds]
        # sorted_measured = flat_measured[sorted_inds]
        #
        # # find the binned averages for predicted and measured responses
        # nbin_width = int(len(sorted_measured)/nbins)
        # pred_binmeans = np.zeros((nbins,1))
        # measured_binmeans = np.zeros((nbins,1))
        # for bin in range(0,nbins):
        #     measured_binmeans[bin] = np.mean(sorted_measured[nbin_width*bin:nbin_width*(bin+1)])
        #     pred_binmeans[bin] = np.mean(sorted_pred[nbin_width*bin:nbin_width*(bin+1)])
        #
        # # try to fit a sigmoid to the data cloud defined by (predicted,measured); if no fit can be found due to the data's shape, fit a line instead
        # try:
        #     xdata = sorted_pred
        #     ydata = sorted_measured
        #     p0 = [np.max(ydata), 0, 1, 0] #initial conditions for gradient descent
        #     popt, pcov = curve_fit(sigmoid, xdata, ydata, p0, method='lm')
        #     all_uv_nl_fits[:,0,roi_ind] = popt
        # except:
        #     xdata = sorted_pred
        #     ydata = sorted_measured
        #     p0 = [1, 0] #initial conditions for gradient descent
        #     popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
        #     all_uv_nl_fits[2:4,0,roi_ind] = popt
        #
        # # repeat these calculations using the 'bounding box' method, subsampling the STRF 5 samples in all directions from the previously calculated centroid (if the edge of the array is reached in any direction, truncate the size of the bounding box)
        # joint_pred = np.mean(ds_preds[np.max([int(cent_inds[0,roi_ind])-4,0]): np.min([int(cent_inds[0,roi_ind])+4,16]),np.max([int(cent_inds[1,roi_ind])-4,0]): np.min([int(cent_inds[1,roi_ind])+4,16]),:,:],axis=(0,1))
        #
        # flat_pred = joint_pred.flatten()
        # flat_measured = ms_measured.flatten()
        # sorted_inds = np.argsort(flat_pred, axis=None)
        # bb_sorted_pred = flat_pred[sorted_inds]
        # bb_sorted_measured = flat_measured[sorted_inds]
        # bb_pred_binmeans = np.zeros((nbins,1))
        # bb_measured_binmeans = np.zeros((nbins,1))
        # for bin in range(0,nbins):
        #     bb_measured_binmeans[bin] = np.mean(bb_sorted_measured[nbin_width*bin:nbin_width*(bin+1)])
        #     bb_pred_binmeans[bin] = np.mean(bb_sorted_pred[nbin_width*bin:nbin_width*(bin+1)])
        #
        # # try to fit a sigmoid to the data cloud defined by (predicted,measured); if no fit can be found due to the data's shape, fit a line instead
        # try:
        #     xdata = bb_sorted_pred
        #     ydata = bb_sorted_measured
        #     p0 = [np.max(ydata), 0, 1, 0] #initial conditions for gradient descent
        #     bb_popt, bb_pcov = curve_fit(sigmoid, xdata, ydata, p0, method='lm')
        #     all_uv_nl_fits[:,1,roi_ind] = bb_popt
        # except:
        #     xdata = bb_sorted_pred
        #     ydata = bb_sorted_measured
        #     p0 = [1, 0] #initial conditions for gradient descent
        #     bb_popt, bb_pcov = curve_fit(line, xdata, ydata, p0, method='lm')
        #     all_uv_nl_fits[2:4,1,roi_ind] = bb_popt
        #
        # # output variables are: mean_nl, a 3D array with 1st dimension 'bin', 2nd dimension holding the predicted and measured mean values, and 3rd dimension 'prediction type'; and sig_params, a 2D array with 1st dimension 'parameter' including L (curve height), x0 (curve midpoint), k (slope) and b (vertical bias) for the sigmoid function, and 2nd dimension 'prediction type'. To recover the nl curves, plot mean_nl[:,1,n] against mean_nl[:,0,n]. To recover the fit curve, redefine the sigmoid function in the top block
        # mean_nl = np.zeros(all_uv_mean_nl.shape[0:3])
        # mean_nl[:,:,0] = np.append(measured_binmeans,pred_binmeans,axis=1)
        # mean_nl[:,:,1] = np.append(bb_measured_binmeans,bb_pred_binmeans,axis=1)
        # all_uv_mean_nl[:,:,:,roi_ind] = mean_nl

# save mean, dc-subtracted, z-scored STRF, and nonlinearity summaries
np.save('/Volumes/TBD/Bruker/STRFs/compiled/all_uv_rois.npy', all_uv_resp)
np.save('/Volumes/TBD/Bruker/STRFs/compiled/all_uv_rois_dff.npy', all_uv_resp_dff)
# np.save('/Volumes/TBD/Bruker/STRFs/compiled/all_uv_mean_NLs.npy', all_uv_mean_nl)
# np.save('/Volumes/TBD/Bruker/STRFs/compiled/all_uv_NL_fits.npy', all_uv_nl_fits)


# %% calculate and save all flicker responses
for roi_ind in range(0, np.shape(scraped_log)[0]):
    # check if the curent ROI exists
    if scraped_log[roi_ind,10] != '':
        # grab important info from scraped log and open ID object
        file_path = os.path.join(experiment_file_directory, scraped_log[roi_ind,2] + '.hdf5')
        series_number = int(float(scraped_log[roi_ind,9]))
        roi_set_name = scraped_log[roi_ind,10]
        roi_num = int(float(scraped_log[roi_ind,11]))
        ID = imaging_data.ImagingDataObject(file_path, series_number, quiet=True)
        # populate roi_data, find unique conditions, and initialize ROI flicker data array
        roi_data = ID.getRoiResponses(roi_set_name)
        unique_parameter_values = np.unique([ep.get('current_temporal_frequency') for ep in ID.getEpochParameters()])
        roi_responses = np.zeros((len(unique_parameter_values),len(flick_resamp_t)))
        # pull trials for each condition
        for param_ind in range(0,len(unique_parameter_values)):
            query = {'current_temporal_frequency': unique_parameter_values[param_ind]}
            trial_data = shared_analysis.filterTrials(roi_data.get('epoch_response'), ID, query)
            condition_responses = np.zeros((trial_data.shape[1],len(flick_resamp_t)))
            for trial in range(0,trial_data.shape[1]):
                # resample at flick_resample_rate
                condition_responses[trial,:] = np.interp(flick_resamp_t,roi_data.get('time_vector'),trial_data[roi_num,trial,:])

            # add mean condition response to roi response array
            roi_responses[param_ind,:] = np.mean(condition_responses,0)

        # add roi responses to master flicker array
        all_flick_resp[:,:,roi_ind] = roi_responses

# save
np.save('/Volumes/TBD/Bruker/STRFs/compiled/all_flick_rois.npy', all_flick_resp)
