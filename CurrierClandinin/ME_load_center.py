from visanalysis.analysis import imaging_data, shared_analysis
from tac_util import tac_h5_tools
import numpy as np
import os
import sys
import cv2
import xlrd
import warnings
from pathlib import Path
from datetime import datetime
from scipy import ndimage
from skimage.measure import regionprops

book = xlrd.open_workbook('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/ME0708_full_log_snap.xls')
sheet = book.sheet_by_name('Sheet1')
data = [[sheet.cell_value(r, c) for c in range(sheet.ncols)] for r in range(sheet.nrows)]
full_log = np.asarray(data)

# reduce full log to rows containing H5-tagged ROIs
H5_bool = full_log[:,0] == 'o'
scraped_log = full_log[H5_bool,:]

# repopulate date strings from excel sequential day format
for roi_ind in range(0, np.shape(scraped_log)[0]):
    excel_date = int(float(scraped_log[roi_ind,2]))
    dt = datetime.fromordinal(datetime(1900, 1, 1).toordinal() + excel_date - 2)
    scraped_log[roi_ind,2] = str(dt)[0:10]

# open compiled response arrays: all_blue_resp, all_uv_resp, all_flick_resp
all_blue_resp = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all_blue_rois.npy')
all_uv_resp = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all_uv_rois.npy')

# %% center all STRFs - only needs to be run once; after, load all_centered_STRFs.npy

# create a single variable that will contain all STRF data, with dimensions (x, y, t, c, roi_ind)
all_centered_STRFs = np.full((80,80,60,2,np.shape(scraped_log)[0]), np.NaN, dtype='float')
all_centered_cores = np.full((80,80,2,np.shape(scraped_log)[0]), np.NaN, dtype='float')
all_centered_surrounds = np.full((80,80,2,np.shape(scraped_log)[0]), np.NaN, dtype='float')
all_rotated_STRFs = np.full((80,80,60,2,np.shape(scraped_log)[0]), np.NaN, dtype='float')
all_rotated_cores = np.full((80,80,2,np.shape(scraped_log)[0]), np.NaN, dtype='float')
all_rotated_surrounds = np.full((80,80,2,np.shape(scraped_log)[0]), np.NaN, dtype='float')
all_mean_peak_SRFs = np.full((80,80,2,np.shape(scraped_log)[0]), np.NaN, dtype='float')

# create cross-neuron summary metric variables
strict_areas = np.full(np.shape(scraped_log)[0], np.NaN, dtype='float')
ellipse_areas = np.full(np.shape(scraped_log)[0], np.NaN, dtype='float')
long_axes = np.full(np.shape(scraped_log)[0], np.NaN, dtype='float')
short_axes = np.full(np.shape(scraped_log)[0], np.NaN, dtype='float')
orient_selects = np.full(np.shape(scraped_log)[0], np.NaN, dtype='float')
pref_orientations = np.full(np.shape(scraped_log)[0], np.NaN, dtype='float')
all_shifts = np.full((3,np.shape(scraped_log)[0]), np.NaN, dtype='float') # X,Y,theta shift to center and rotate STRFs

# generate along-axis indexing matrices for horiz, vert, diag, and anti-diag. create one matrix on main axis, and one shifted 5º (1 spatial sample) above and below main axis. these will be used to determine the principle axis for rotational alignment
horiz_index_mats = np.zeros((80,80,3))
vert_index_mats = np.zeros((80,80,3))
diag_index_mats = np.zeros((80,80,3))
antidiag_index_mats = np.zeros((80,80,3))
for n in range(0,horiz_index_mats.shape[2]):
    horiz_index_mats[35+(5*n),:,n].fill(1)
    vert_index_mats[:,35+(5*n),n].fill(1)
np.fill_diagonal(diag_index_mats[:,:,1],1)
diag_index_mats[:,:,1] = np.roll(diag_index_mats[:,:,1],1,axis=0)
diag_index_mats[:,:,0] = np.roll(diag_index_mats[:,:,1],-5,axis=0)
diag_index_mats[:,:,2] = np.roll(diag_index_mats[:,:,1],5,axis=0)
antidiag_index_mats = np.flipud(diag_index_mats)

# for each index in the scraped log, pull up the blue and UV STRFs, then center
# for roi_ind in range(331, 332):
for roi_ind in range(0, np.shape(scraped_log)[0]):
    # load blue and uv STRFs for ROI
    roi_strfs = np.zeros(all_blue_resp.shape[0:3]+(2,))
    roi_strfs[:,:,:,0] = all_blue_resp[:,:,:,roi_ind]
    roi_strfs[:,:,:,1] = all_uv_resp[:,:,:,roi_ind]
    # find CoM for blue and UV
    com=np.zeros((2,2))
    centering_core = np.full(all_blue_resp.shape[0:2]+(2,), np.NaN, dtype='float')
    surr_core = np.full(all_blue_resp.shape[0:2]+(2,), np.NaN, dtype='float')
    for c in range(0,roi_strfs.shape[3]):
        if scraped_log[roi_ind,4+int(c*3)] != '':
            current_strf = roi_strfs[:,:,:,c]
            # since all STRFs are the same shape (15360 for 3 sec filter), the bottom and top 2.5% boundaries will always be defined by the same index in the sorted flattened array
            # there may be weird behavior for some STRFs near noise level depending on the indices used here. 0.5% = 77/15283; 1.25% = 192/15168; 2.5% = 384/14976; 5% = 768/14592
            minthresh = np.sort(np.ndarray.flatten(current_strf))[77]
            maxthresh = np.sort(np.ndarray.flatten(current_strf))[15283]
            # apply mask to entire STRF based on previously defined thresholds
            minmasked = np.where(current_strf<=minthresh,current_strf,0)
            maxmasked = np.where(current_strf>=maxthresh,current_strf,0)
            # take mean masked srf of last 0.5 sec - catches final peak of biphasic TRFs
            minm_srf = np.average(minmasked[:,:,50:],axis=2)
            maxm_srf = np.average(maxmasked[:,:,50:],axis=2)
            # find largest contiguous patch in mean mask
            min_features = ndimage.label(minm_srf, structure=np.full((3,3),1))[0]
            max_features = ndimage.label(maxm_srf, structure=np.full((3,3),1))[0]
            ## ITERATE OVER FEATURES TO FIND THE ONE WITH THE LARGEST SUM, INSTEAD OF JUST THE LARGEST AREA. THIS ACCOUNTS FOR BOTH SIZE OF FEATURE AND ITS INTENSITY
            min_feat_weights = np.zeros(np.amax(min_features))
            for f in range(1,np.amax(min_features)+1):
                test_centroid = np.where(min_features==f,minm_srf,np.NaN)
                min_feat_weights[f-1] = np.nansum(test_centroid,(0,1))
            min_label = np.where(min_feat_weights==np.amin(min_feat_weights))[0][0]+1
            min_centroid = np.where(min_features==min_label,minm_srf,np.NaN)

            max_feat_weights = np.zeros(np.amax(max_features))
            for f in range(1,np.amax(max_features)+1):
                test_centroid = np.where(max_features==f,maxm_srf,np.NaN)
                max_feat_weights[f-1] = np.nansum(test_centroid,(0,1))
            max_label = np.where(max_feat_weights==np.amax(max_feat_weights))[0][0]+1
            max_centroid = np.where(max_features==max_label,maxm_srf,np.NaN)
            # determine whether to use positive or negative for centering based on which centroid has a larger sum. the surround is defined as the strongest feature with the opposite sign of the center (might need to rethink this)
            if np.abs(np.nansum(min_centroid,(0,1))) > np.nansum(max_centroid,(0,1)):
                centering_core[:,:,c] = min_centroid
            else:
                centering_core[:,:,c] = max_centroid
            # weight array indices by centroid value and find mean - this is the center of mass
            zero_core = np.where(np.isnan(centering_core[:,:,c]),0,centering_core[:,:,c])
            ind_array = np.nonzero(zero_core)
            com[1,c] = np.average(ind_array[0], axis=0, weights=np.abs(zero_core[ind_array]))
            com[0,c] = np.average(ind_array[1], axis=0, weights=np.abs(zero_core[ind_array]))

    # run again with less stringent threshold for surr_core definition
    for c in range(0,roi_strfs.shape[3]):
        if scraped_log[roi_ind,4+int(c*3)] != '':
            current_strf = roi_strfs[:,:,:,c]
            # 0.5% = 77/15283; 1.25% = 192/15168; 2.5% = 384/14976; 5% = 768/14592
            minthresh = np.sort(np.ndarray.flatten(current_strf))[768]
            maxthresh = np.sort(np.ndarray.flatten(current_strf))[14592]
            # apply mask to entire STRF based on previously defined thresholds
            minmasked = np.where(current_strf<=minthresh,current_strf,0)
            maxmasked = np.where(current_strf>=maxthresh,current_strf,0)
            # take mean masked srf of last 0.5 sec - catches final peak of biphasic TRFs
            minm_srf = np.average(minmasked[:,:,50:],axis=2)
            maxm_srf = np.average(maxmasked[:,:,50:],axis=2)
            # find largest contiguous patch in mean mask
            min_features = ndimage.label(minm_srf, structure=np.full((3,3),1))[0]
            max_features = ndimage.label(maxm_srf, structure=np.full((3,3),1))[0]
            if np.nansum(centering_core[:,:,c]) > 0:
                min_feat_weights = np.zeros(np.amax(min_features))
                for f in range(1,np.amax(min_features)+1):
                    test_centroid = np.where(min_features==f,minm_srf,np.NaN)
                    min_feat_weights[f-1] = np.nansum(test_centroid,(0,1))
                min_label = np.where(min_feat_weights==np.amin(min_feat_weights))[0][0]+1
                surr_core[:,:,c] = np.where(min_features==min_label,minm_srf,np.NaN)
            else:
                max_feat_weights = np.zeros(np.amax(max_features))
                for f in range(1,np.amax(max_features)+1):
                    test_centroid = np.where(max_features==f,maxm_srf,np.NaN)
                    max_feat_weights[f-1] = np.nansum(test_centroid,(0,1))
                max_label = np.where(max_feat_weights==np.amax(max_feat_weights))[0][0]+1
                surr_core[:,:,c] = np.where(max_features==max_label,maxm_srf,np.NaN)

    # master_com is the CoM from the color with the largest sum
    if np.abs(np.nansum(centering_core[:,:,0],axis=(0,1))) > np.abs(np.nansum(centering_core[:,:,1],axis=(0,1))):
        master_com = np.around(com[:,0] * 5 + 2)
    else:
        master_com = np.around(com[:,1] * 5 + 2)
    # calculate number of indices to shift in x and y based on the overall CoM
    ind_shift = [40-int(master_com[0]), 40-int(master_com[1])]
    # upsample both STRFs in x and y by a factor of 5, so that each pixel is 1º
    ups_strf = np.repeat(np.repeat(roi_strfs,5,axis=0),5,axis=1)
    # create padded 0 array and populate with shifted STRF - using zeros here ensures that "empty" parts of centered STRFs are 0, while fully missing STRFs remain NaNs
    padded_strf = (np.tile(np.zeros(ups_strf.shape, dtype='float'),(3,3,1,1)))
    padded_strf[80+ind_shift[1]:160+ind_shift[1],80+ind_shift[0]:160+ind_shift[0],:,:] = ups_strf
    # index center of padded array to return shifted STRF
    all_centered_STRFs[:,:,:,:,roi_ind] = padded_strf[80:160,80:160,:,:]
    # write shift params to table
    all_shifts[0:2,roi_ind] = ind_shift

    # upsample centering core in x and y by a factor of 5, so that each pixel is 1º
    ups_core = np.repeat(np.repeat(centering_core,5,axis=0),5,axis=1)
    # create padded 0 array and populate with shifted core
    padded_core = (np.tile(np.zeros(ups_core.shape, dtype='float'),(3,3,1)))
    padded_core[80+ind_shift[1]:160+ind_shift[1],80+ind_shift[0]:160+ind_shift[0],:] = ups_core
    # index center of padded core to return shifted core
    all_centered_cores[:,:,:,roi_ind] = padded_core[80:160,80:160,:]

    # upsample surround core in x and y by a factor of 5, so that each pixel is 1º
    ups_core = np.repeat(np.repeat(surr_core,5,axis=0),5,axis=1)
    # create padded 0 array and populate with shifted core
    padded_core = (np.tile(np.zeros(ups_core.shape, dtype='float'),(3,3,1)))
    padded_core[80+ind_shift[1]:160+ind_shift[1],80+ind_shift[0]:160+ind_shift[0],:] = ups_core
    # index center of padded core to return shifted core
    all_centered_surrounds[:,:,:,roi_ind] = padded_core[80:160,80:160,:]

    # convert output variable NaNs to zeros and mask surround to avoid including center indices
    all_centered_cores[:,:,:,roi_ind] = np.where(np.isnan(all_centered_cores[:,:,:,roi_ind]),0,all_centered_cores[:,:,:,roi_ind])
    all_centered_surrounds[:,:,:,roi_ind] = np.where(np.isnan(all_centered_surrounds[:,:,:,roi_ind]),0,all_centered_surrounds[:,:,:,roi_ind])
    all_centered_surrounds[:,:,:,roi_ind] = np.where(all_centered_cores[:,:,:,roi_ind] == 0, all_centered_surrounds[:,:,:,roi_ind], 0)


    # STRF rotation - general idea is to find the maximal sum of the centering core along each cardinal and off-cardinal axis, then transform the STRF matrix so that the maximal sum axis is oriented in a consistent way (i.e., horizontally along the x-axis)
    # grab centered core for color with strongest filter to use for rotation calcs
    if np.abs(np.nansum(centering_core[:,:,0],axis=(0,1))) > np.abs(np.nansum(centering_core[:,:,1],axis=(0,1))):
        centered_core = all_centered_cores[:,:,0,roi_ind]
    else:
        centered_core = all_centered_cores[:,:,1,roi_ind]

    # convert NaNs to zeros and take absolute value to ease future computations
    centered_core = np.where(np.isnan(centered_core),0,centered_core)
    centered_core = np.abs(centered_core)

    # apply each indexing matrix to centered core and sum over indexed values (summing accounts for both spatial extent AND filter strength, instead of only one or the other)
    h_weights = np.zeros(horiz_index_mats.shape[2])
    v_weights = np.zeros(horiz_index_mats.shape[2])
    d_weights = np.zeros(horiz_index_mats.shape[2])
    ad_weights = np.zeros(horiz_index_mats.shape[2])
    h_extents = np.zeros(horiz_index_mats.shape[2])
    v_extents = np.zeros(horiz_index_mats.shape[2])
    d_extents = np.zeros(horiz_index_mats.shape[2])
    ad_extents = np.zeros(horiz_index_mats.shape[2])
    for n in range(0,horiz_index_mats.shape[2]):
        h_weights[n] = np.sum(centered_core[horiz_index_mats[:,:,n]==1])
        h_extents[n] = len(np.where(centered_core[horiz_index_mats[:,:,n]==1])[0])
        v_weights[n] = np.sum(centered_core[vert_index_mats[:,:,n]==1])
        v_extents[n] = len(np.where(centered_core[vert_index_mats[:,:,n]==1])[0])
        d_weights[n] = np.sum(centered_core[diag_index_mats[:,:,n]==1])
        d_extents[n] = len(np.where(centered_core[diag_index_mats[:,:,n]==1])[0])
        ad_weights[n] = np.sum(centered_core[antidiag_index_mats[:,:,n]==1])
        ad_extents[n] = len(np.where(centered_core[antidiag_index_mats[:,:,n]==1])[0])

    # find max difference between D-AD and H-V weights; largest difference = most selective
    hv_diff = np.amax([np.amax(v_weights)-np.amax(h_weights), np.amax(h_weights)-np.amax(v_weights)])
    dad_diff = np.amax([np.amax(d_weights)-np.amax(ad_weights), np.amax(ad_weights)-np.amax(d_weights)])

    # within the H-V or D-AD axes, find largest weight and apply appropriate transform
    if hv_diff >= dad_diff:
        # area in square degrees is the product of pi * biggest long axis/2 * median short axis/2
        # this models each SRF as an ellipse, and treats the long and short axes as radii
        # cannot use the shortest short axis, because it will often be 0 for small spatial RFs; median is a better representation than mean of the short axis
        if np.amax(h_weights) < np.amax(v_weights):
            weight_alignment_shift = (np.where(v_weights==np.amax(v_weights))[0][0] * 5)-5
            # vertical shift means spatial plane is rotated by 90º
            all_rotated_cores[:,:,:,roi_ind] = np.roll(np.rot90(all_centered_cores[:,:,:,roi_ind]),weight_alignment_shift,axis=0)
            all_rotated_surrounds[:,:,:,roi_ind] = np.roll(np.rot90(all_centered_surrounds[:,:,:,roi_ind]),weight_alignment_shift,axis=0)
            all_rotated_STRFs[:,:,:,:,roi_ind] = np.roll(np.rot90(all_centered_STRFs[:,:,:,:,roi_ind]),weight_alignment_shift,axis=0)
            # write rotation to table
            all_shifts[2,roi_ind] = 90
        else:
            # this will be the outcome if all weights are equal - no rotation
            # horizontal shift means no rotation is required, but can still roll according to max weight index
            weight_alignment_shift = (np.where(h_weights==np.amax(h_weights))[0][0] * 5)-5
            all_rotated_cores[:,:,:,roi_ind] = np.roll(all_centered_cores[:,:,:,roi_ind],weight_alignment_shift,axis=0)
            all_rotated_surrounds[:,:,:,roi_ind] = np.roll(all_centered_surrounds[:,:,:,roi_ind],weight_alignment_shift,axis=0)
            all_rotated_STRFs[:,:,:,:,roi_ind] = np.roll(all_centered_STRFs[:,:,:,:,roi_ind],weight_alignment_shift,axis=0)
            all_shifts[2,roi_ind] = 0
    else:
        if np.amax(d_weights) < np.amax(ad_weights):
            # antidiagonal shift
            weight_alignment_shift = (np.where(ad_weights==np.amax(ad_weights))[0][0] * 5)-5
            all_rotated_cores[:,:,:,roi_ind] = np.roll(ndimage.rotate(all_centered_cores[:,:,:,roi_ind],135,reshape=False), weight_alignment_shift,axis=0)
            all_rotated_surrounds[:,:,:,roi_ind] = np.roll(ndimage.rotate(all_centered_surrounds[:,:,:,roi_ind],135,reshape=False), weight_alignment_shift,axis=0)
            all_rotated_STRFs[:,:,:,:,roi_ind] = np.roll(ndimage.rotate(all_centered_STRFs[:,:,:,:,roi_ind],135,reshape=False), weight_alignment_shift,axis=0)
            all_shifts[2,roi_ind] = 135

        else:
            # diagonal shift
            weight_alignment_shift = (np.where(d_weights==np.amax(d_weights))[0][0] * 5)-5
            all_rotated_cores[:,:,:,roi_ind] = np.roll(ndimage.rotate(all_centered_cores[:,:,:,roi_ind],45,reshape=False), weight_alignment_shift,axis=0)
            all_rotated_surrounds[:,:,:,roi_ind] = np.roll(ndimage.rotate(all_centered_surrounds[:,:,:,roi_ind],45,reshape=False), weight_alignment_shift,axis=0)
            all_rotated_STRFs[:,:,:,:,roi_ind] = np.roll(ndimage.rotate(all_centered_STRFs[:,:,:,:,roi_ind],45,reshape=False), weight_alignment_shift,axis=0)
            all_shifts[2,roi_ind] = 45

    # STRF SPATIAL STATISTICS - overall idea is to repeat feature detection on a more precise temporal snippet of STRF that has been thresholded by an absolute z value (rather than an STRF-relative z value, as above). what looks like "background/noise" visually falls in the interval [-0.65,0.65]; STRF elements that do not cross this threshold cannot be unambiguously assigned to the cell's response. performing this secondary thresholding improves the calculation of spatial statistics by rejecting STRF elements that live below the noise level
    if np.abs(np.nansum(centering_core[:,:,0],axis=(0,1))) > np.abs(np.nansum(centering_core[:,:,1],axis=(0,1))):
        useblue = True
        # grab the TRF of the CoM of the color that was used for centering
        com_trf = all_centered_STRFs[39,39,:,0,roi_ind]
        # find the peak of the CoM TRF
        peakind = np.argmax(np.abs(com_trf[50:]))+50
        # find last zero crossings before and after peakind
        pre_diff = np.diff(np.sign(com_trf[:peakind]))
        pre_ind = np.max(np.where(pre_diff)) + 1
        try:
            post_diff = np.diff(np.sign(cell_TRFs[peakind:,0]))
            post_ind = peakind + np.min(np.where(post_diff))
        except:
            post_ind = 59
        # take mean SRF for period defined by these indices
        peak_STRF = all_centered_STRFs[:,:,pre_ind:post_ind,0,roi_ind]
        # take nonpeak_STRF from same time snippet for other color
        nonpeak_STRF = all_centered_STRFs[:,:,pre_ind:post_ind,1,roi_ind]
    else:
        useblue = False
        # perform same computations on UV data, if this color was used for original centering
        com_trf = all_centered_STRFs[39,39,:,1,roi_ind]
        peakind = np.argmax(np.abs(com_trf[50:]))+50
        pre_diff = np.diff(np.sign(com_trf[:peakind]))
        pre_ind = np.max(np.where(pre_diff)) + 1
        try:
            post_diff = np.diff(np.sign(cell_TRFs[peakind:,0]))
            post_ind = peakind + np.min(np.where(post_diff))
        except:
            post_ind = 59
        peak_STRF = all_centered_STRFs[:,:,pre_ind:post_ind,1,roi_ind]
        nonpeak_STRF = all_centered_STRFs[:,:,pre_ind:post_ind,0,roi_ind]

    # take average over both colors for spatial calcs
    joint_peak_STRF = (peak_STRF+nonpeak_STRF)/2

    # threshold peak STRF snippet with an ABSOLUTE z value based on overall STRF statistics - sign of threshold is determined by the sign of the peak TRF value
    if np.sign(com_trf[peakind]) > 0:
        threshval = 0.5
        # replace pixels that don't cross threshold with nans, then define a mean SRF that replaces nans with 0s (this two-step operation preserves intensity for final ellipse fitting)
        thresh_peak_STRF = np.where(peak_STRF>threshval,peak_STRF,np.nan)
        thresh_joint_peak_STRF = np.where(joint_peak_STRF>threshval,joint_peak_STRF,np.nan)
    else:
        threshval = -0.5
        thresh_peak_STRF = np.where(peak_STRF<threshval,peak_STRF,np.nan)
        thresh_joint_peak_STRF = np.where(joint_peak_STRF<threshval,joint_peak_STRF,np.nan)
    mean_peak_SRF = np.nanmean(thresh_peak_STRF,axis=2)
    mean_peak_SRF = np.where(np.isnan(mean_peak_SRF),0,mean_peak_SRF)
    mean_joint_SRF = np.nanmean(thresh_joint_peak_STRF,axis=2)
    mean_joint_SRF = np.where(np.isnan(mean_joint_SRF),0,mean_joint_SRF)

    # repeat for nonpeak_STRF
    if useblue:
        if np.sign(all_centered_STRFs[39,39,peakind,1,roi_ind]) > 0:
            threshval = 0.5
            thresh_nonpeak_STRF = np.where(nonpeak_STRF>threshval,nonpeak_STRF,np.nan)
        else:
            threshval = -0.5
            thresh_nonpeak_STRF = np.where(nonpeak_STRF<threshval,nonpeak_STRF,np.nan)
    else:
        if np.sign(all_centered_STRFs[39,39,peakind,0,roi_ind]) > 0:
            threshval = 0.5
            thresh_nonpeak_STRF = np.where(nonpeak_STRF>threshval,nonpeak_STRF,np.nan)
        else:
            threshval = -0.5
            thresh_nonpeak_STRF = np.where(nonpeak_STRF<threshval,nonpeak_STRF,np.nan)
    mean_nonpeak_SRF = np.nanmean(thresh_nonpeak_STRF,axis=2)
    mean_nonpeak_SRF = np.where(np.isnan(mean_nonpeak_SRF),0,mean_nonpeak_SRF)

    # check to make sure STRF passes this secondary thresholding
    if mean_peak_SRF[39,39]==0:
        # if the center of the SRF contains no signal, keep spatial params as nans
        pass
    else:
        # define index of central feature of SRF (lives at 39,39)
        ellip_feats = ndimage.label(mean_peak_SRF, structure=np.full((3,3),1))[0]
        cent_feat = ellip_feats[39,39]
        # fit ellipse model and pull center parameters
        improps = regionprops(np.where(ellip_feats==cent_feat,1,0), intensity_image=np.where(ellip_feats==cent_feat,mean_peak_SRF,0))[0]
        long_axes[roi_ind] = improps['major_axis_length']
        short_axes[roi_ind] = improps['minor_axis_length']
        strict_areas[roi_ind] = improps['area']
        ellipse_areas[roi_ind] = (improps['major_axis_length']/2)*(improps['minor_axis_length']/2)*np.pi
        if useblue:
            all_mean_peak_SRFs[:,:,0,roi_ind] = mean_peak_SRF
            if np.nonzero(mean_nonpeak_SRF[39,39]):
                all_mean_peak_SRFs[:,:,1,roi_ind] = mean_nonpeak_SRF
        else:
            all_mean_peak_SRFs[:,:,1,roi_ind] = mean_peak_SRF
            if np.nonzero(mean_nonpeak_SRF[39,39]):
                all_mean_peak_SRFs[:,:,0,roi_ind] = mean_nonpeak_SRF

    # calculate OSI on color-joint version
    if mean_joint_SRF[39,39]==0:
        # if the center of the SRF contains no signal, keep spatial params as nans
        pass
    else:
        # define index of central feature of SRF (lives at 39,39)
        ellip_feats = ndimage.label(mean_joint_SRF, structure=np.full((3,3),1))[0]
        cent_feat = ellip_feats[39,39]
        # fit ellipse model and pull center parameters
        improps = regionprops(np.where(ellip_feats==cent_feat,1,0), intensity_image=np.where(ellip_feats==cent_feat,mean_joint_SRF,0))[0]
        orient_selects[roi_ind] = improps['major_axis_length']/improps['minor_axis_length']
        pref_orientations[roi_ind] = np.rad2deg(improps['orientation'])


# save centered and rotated STRFs and cores for later loading/analysis
np.save('/Volumes/TBD/Bruker/STRFs/compiled/all_centered_strfs.npy', all_centered_STRFs)
np.save('/Volumes/TBD/Bruker/STRFs/compiled/all_centered_cores.npy', all_centered_cores)
np.save('/Volumes/TBD/Bruker/STRFs/compiled/all_centered_surrounds.npy', all_centered_surrounds)
np.save('/Volumes/TBD/Bruker/STRFs/compiled/all_rotated_strfs.npy', all_rotated_STRFs)
np.save('/Volumes/TBD/Bruker/STRFs/compiled/all_rotated_cores.npy', all_rotated_cores)
np.save('/Volumes/TBD/Bruker/STRFs/compiled/all_rotated_surrounds.npy', all_rotated_surrounds)
np.save('/Volumes/TBD/Bruker/STRFs/compiled/all_orientation_selectivity_index.npy', orient_selects)
np.save('/Volumes/TBD/Bruker/STRFs/compiled/all_SRF_center_strict_areas.npy', strict_areas)
np.save('/Volumes/TBD/Bruker/STRFs/compiled/all_ellipse_model_areas.npy', ellipse_areas)
np.save('/Volumes/TBD/Bruker/STRFs/compiled/all_long_axes.npy', long_axes)
np.save('/Volumes/TBD/Bruker/STRFs/compiled/all_short_axes.npy', short_axes)
np.save('/Volumes/TBD/Bruker/STRFs/compiled/all_pref_orientations.npy', pref_orientations)
np.save('/Volumes/TBD/Bruker/STRFs/compiled/all_shifts.npy', all_shifts)
np.save('/Volumes/TBD/Bruker/STRFs/compiled/all_mean_peak_SRFs.npy', all_mean_peak_SRFs)
