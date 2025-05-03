from visanalysis.analysis import imaging_data, shared_analysis
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import cv2
import xlrd
import csv
import warnings
from pathlib import Path
from datetime import datetime
from scipy import ndimage
from scipy import stats

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

# open compiled response arrays: all_rotated_STRFs, all_flick_resp
all_centered_STRFs = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all_centered_strfs.npy')
all_centered_cores = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all_centered_cores.npy')
all_centered_surrounds = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all_centered_surrounds.npy')
all_rotated_cores = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all_rotated_cores.npy')
all_rotated_surrounds = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all_rotated_surrounds.npy')
all_rotated_STRFs = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all_rotated_strfs.npy')
all_flick_resp = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all_flick_rois.npy')

# flicker time vector and frequencies
flick_resamp_t = np.arange(0, 14-0.8, 1/5)
flick_freqs = [0.1,0.5,1,2]


#%% Plot summary PDFs for each unique cell type in the dataset. Takes about 2 sec per cell type to run

# disable runtime and deprecation warnings - dangerous! turn this off while working on the function
warnings.filterwarnings("ignore")

# pull list of unique cell type labels
utype_indices = np.unique(scraped_log[:,14]) != ''
unique_types = np.unique(scraped_log[:,14])[utype_indices]

# create empty array for mean flattened responses
mean_small_flat_resp = np.zeros((len(unique_types), len(np.ravel(all_centered_STRFs[25:55,25:55,40:,:,0])) + len(np.ravel(all_flick_resp[:,:,0]))), dtype='float')

# for ind in range(57,65):
for ind in range(0,len(unique_types)):
    # define cell type to summarize
    label = unique_types[ind]
    # grab certainty values for each N
    label_certs = scraped_log[scraped_log[:,14] == label,15].astype('float')
    # grab responses at indices where label matches cell type column
    label_STRFs = all_centered_STRFs[:,:,:,:,scraped_log[:,14] == label]
    n_cells = label_STRFs.shape[4]

    # defining TRF based on argmax of centered core (peak responding pixel in STRF center)
    label_cores = all_centered_cores[:,:,:,scraped_log[:,14] == label]
    label_surround_masks = all_centered_surrounds[:,:,:,scraped_log[:,14] == label]
    label_TRFs = np.zeros(label_STRFs.shape[2:5])
    label_surround_TRFs = np.zeros(label_STRFs.shape[2:5])
    label_inner_TRFs = np.zeros((label_TRFs.shape[0],label_TRFs.shape[2]))
    label_surround_inner_TRFs = np.zeros((label_TRFs.shape[0],label_TRFs.shape[2]))
    # grab the larger weight core that was actually used for centering, and find max
    for cell in range(0,n_cells):
        if np.abs(np.nansum(label_cores[:,:,0,cell],axis=(0,1))) > np.abs(np.nansum(label_cores[:,:,1,cell],axis=(0,1))):
            current_core = label_cores[:,:,0,cell].flatten()
            SRF_center_ind = np.argmax(np.abs(current_core))
            # take full timecourse for the max responding pixel
            # Note that using this method impairs the SRF if max responding pixels are not already well aligned!
            label_TRFs[:,0,cell] = label_STRFs.reshape(-1,60,2,n_cells)[SRF_center_ind,:,0,cell]
            label_TRFs[:,1,cell] = label_STRFs.reshape(-1,60,2,n_cells)[SRF_center_ind,:,1,cell]
        else:
            current_core = label_cores[:,:,1,cell].flatten()
            SRF_center_ind = np.argmax(np.abs(current_core))
            label_TRFs[:,0,cell] = label_STRFs.reshape(-1,60,2,n_cells)[SRF_center_ind,:,0,cell]
            label_TRFs[:,1,cell] = label_STRFs.reshape(-1,60,2,n_cells)[SRF_center_ind,:,1,cell]
        # use the surround mask to grab mean STRF activity over time in the region defined by the mask (for each color and cell)
        label_surround_TRFs[:,0,cell] = np.mean(label_STRFs[np.abs(label_surround_masks[:,:,0,cell]) > 0,:,0,cell],axis=0)
        label_surround_TRFs[:,1,cell] = np.mean(label_STRFs[np.abs(label_surround_masks[:,:,1,cell]) > 0,:,1,cell],axis=0)
        # inner PR contribution to center and surround TRFs is defined as UV-blue for each spatial segment
        label_inner_TRFs[:,cell] = label_TRFs[:,1,cell] - label_TRFs[:,0,cell]
        label_surround_inner_TRFs[:,cell] = label_surround_TRFs[:,1,cell] - label_surround_TRFs[:,0,cell]

    # label flicker responses
    label_flicks = all_flick_resp[:,:,scraped_log[:,14] == label]

    # find latest zero crossing of center for each TRF
    blue_SRFs = np.full((label_STRFs.shape[0],label_STRFs.shape[1],label_STRFs.shape[4]), np.NaN, dtype='float')
    uv_SRFs = np.full((label_STRFs.shape[0],label_STRFs.shape[1],label_STRFs.shape[4]), np.NaN, dtype='float')
    for cell in range(0,n_cells):
        # find TRF index with largest value in each cell
        if not np.isnan(np.mean(label_TRFs[:,0,:],axis=0))[cell]:
            # find peak over period used for centering (last 0.5 sec)
            peakind = np.argmax(np.abs(label_TRFs[50:,0,cell]))+50
            # find last zero crossings before and after peakind
            pre_diff = np.diff(np.sign(label_TRFs[:peakind,0,cell]))
            pre_ind = np.max(np.where(pre_diff)) + 1
            try:
                post_diff = np.diff(np.sign(label_TRFs[peakind:,0,cell]))
                post_ind = peakind + np.min(np.where(post_diff))
            except:
                post_ind = 59
            # take mean SRF for period defined by these indices
            cell_blue_SRF = np.nanmean(label_STRFs[:,:,pre_ind:post_ind,0,cell],axis=2)
            blue_SRFs[:,:,cell] = cell_blue_SRF

        # perform same operation for uv SRF
        if not np.isnan(np.mean(label_TRFs[:,1,:],axis=0))[cell]:
            # find peak over period used for centering (last 0.5 sec)
            peakind = np.argmax(np.abs(label_TRFs[50:,1,cell]))+50
            # find last zero crossings before and after peakind
            pre_diff = np.diff(np.sign(label_TRFs[:peakind,1,cell]))
            pre_ind = np.max(np.where(pre_diff)) + 1
            try:
                post_diff = np.diff(np.sign(label_TRFs[peakind:,1,cell]))
                post_ind = peakind + np.min(np.where(post_diff))
            except:
                post_ind = 59
            # take SRF for period defined by these indices
            cell_uv_SRF = np.nanmean(label_STRFs[:,:,pre_ind:post_ind,1,cell],axis=2)
            uv_SRFs[:,:,cell] = cell_uv_SRF

    # calculate mean responses and std
    mean_STRF = np.nanmean(label_STRFs, axis=4)
    STRF_std = np.nanstd(label_STRFs, axis=4)
    mean_flick = np.nanmean(label_flicks, axis=2)
    flick_sem = np.nanstd(label_flicks, axis=2)/np.sqrt(n_cells)

    # calculate summary figures
    cent_TRF = np.nanmean(label_TRFs,axis=2)
    cent_TRF_sem = np.nanstd(label_TRFs,axis=2)/np.sqrt(n_cells)
    surr_TRF = np.nanmean(label_surround_TRFs,axis=2)
    surr_TRF_sem = np.nanstd(label_surround_TRFs,axis=2)/np.sqrt(n_cells)
    inner_TRF = np.nanmean(label_inner_TRFs,axis=1)
    inner_TRF_sem = np.nanstd(label_inner_TRFs,axis=1)/np.sqrt(n_cells)
    surr_inner_TRF = np.nanmean(label_surround_inner_TRFs,axis=1)
    surr_inner_TRF_sem = np.nanstd(label_surround_inner_TRFs,axis=1)/np.sqrt(n_cells)
    blue_SRF = np.nanmean(blue_SRFs,axis=2)
    uv_SRF = np.nanmean(uv_SRFs,axis=2)
    # determine if mean response contains any NaN
    if not np.any(np.append(np.isnan(mean_STRF),np.isnan(mean_flick))):
        # flatten and spatially sub-sample mean STRF
        small_flat_STRF = np.ravel(mean_STRF[25:55,25:55,40:,:])
        flat_flicks = np.ravel(mean_flick)
        # add to mean flat response arrays
        mean_small_flat_resp[ind,:] = np.append(small_flat_STRF,flat_flicks)

    # plot section
    sumfig = plt.figure(figsize=(15, 10))
    subfigs = sumfig.subfigures(3,1,height_ratios=[1.4,0.85,2.3])
    # blue STRF center
    axs0 = subfigs[0].subplots(2, 11)
    tp = np.arange(50,60,1)

    # rescale plots relative to peak
    rangemax = np.max([np.round(np.max(np.abs(mean_STRF[20:60,20:60,:,:])),1),1])

    for n in range(0,10):
        x = plt.subplot(2,11,(n+1));
        plt.imshow(mean_STRF[20:60,20:60,int(tp[n]),0]/rangemax, origin='lower', cmap='PiYG', clim=[-1,1]);
        plt.title(str(np.round((-3+((tp[n]))/20),2))+' s')
        x.axes.get_xaxis().set_ticks([])
        if n != 0:
            x.axes.get_yaxis().set_ticks([])
        else:
            plt.ylabel('Blue')
            x.axes.get_yaxis().set_ticks([0,40])
            x.axes.get_yaxis().set_ticklabels(['20','60'])
    # cell type info
    x = plt.subplot(2,11,11);
    plt.axis('off')
    plt.text(-0.075,0.65,label, fontsize=25, fontweight='black')
    plt.text(-0.075,0.35,str(len(np.unique(scraped_log[scraped_log[:,14] == label,1])))+' cells', fontsize='large', fontweight='bold')
    plt.text(-0.075,0.15,str(n_cells)+' ROIs', fontsize='large', fontweight='bold')

    # UV STRF center
    for n in range(0,10):
        x = plt.subplot(2,11,(n+12));
        plt.imshow(mean_STRF[20:60,20:60,int(tp[n]),1]/rangemax, origin='lower', cmap='PiYG', clim=[-1,1]);
        x.axes.get_xaxis().set_ticks([0,40])
        x.axes.get_xaxis().set_ticklabels(['20','60'])
        if n != 0:
            x.axes.get_yaxis().set_ticks([])
        else:
            plt.ylabel('UV')
            x.axes.get_yaxis().set_ticks([0,40])
            x.axes.get_yaxis().set_ticklabels(['20','60'])
    # colorbar
    x = plt.subplot(2,11,22);
    plt.axis('off')
    plt.colorbar(fraction=0.8, aspect=8, label='Norm. Resp.', shrink=1);

    # flicker responses - needed to add a try block because some cell types contain no flicker data
    try:
        axs1 = subfigs[1].subplots(1, 4)
        flick_freqs = [0.1,0.5,1,2]
        rangemax = np.max([np.round(np.max(mean_flick)+np.max(flick_sem),1)+0.2,1])
        rangemin = np.min([np.round(np.min(mean_flick)-np.max(flick_sem),1)-0.2,-0.5])
        for n in range(0,4):
            x = plt.subplot(1,4,(n+1));
            plt.plot(flick_resamp_t,mean_flick[n,:],'k')
            plt.plot(flick_resamp_t,(mean_flick[n,:]+flick_sem[n,:]),'k',linewidth=0.5)
            plt.plot(flick_resamp_t,(mean_flick[n,:]-flick_sem[n,:]),'k',linewidth=0.5)
            plt.title(str(flick_freqs[n])+' Hz')
            if n != 0:
                x.axes.get_yaxis().set_ticks([])
            else:
                plt.ylabel('Mean Flicker dF/F')
            x.axes.get_xaxis().set_ticks([2,7,12])
            x.axes.get_xaxis().set_ticklabels(['0','5','10'])
            plt.ylim([rangemin,rangemax])
    except:
        pass

    # center TRFs
    axs2 = subfigs[2].subplots(2,5)
    ax = plt.subplot(2,5,1)
    rangemax = np.max([np.round(np.max(np.abs(cent_TRF)),1)+np.round(np.max(cent_TRF_sem),1)+0.2,1])
    plt.plot(np.arange(-3,0,1/20),cent_TRF[:,0],color=[0.1,0.2,0.8])
    plt.plot(np.arange(-3,0,1/20),(cent_TRF[:,0]+cent_TRF_sem[:,0]), color=[0.1,0.2,0.8],linewidth=0.5)
    plt.plot(np.arange(-3,0,1/20),(cent_TRF[:,0]-cent_TRF_sem[:,0]),color=[0.1,0.2,0.8],linewidth=0.5)
    plt.plot([-3,0],[0,0],'k:')
    plt.title('Blue')
    plt.ylabel('Centers')
    plt.ylim([-1*rangemax,rangemax])
    ax.axes.get_xaxis().set_ticks([])
    ax = plt.subplot(2,5,2)
    plt.plot(np.arange(-3,0,1/20),cent_TRF[:,1],color=[1,0,1])
    plt.plot(np.arange(-3,0,1/20),(cent_TRF[:,1]+cent_TRF_sem[:,1]),color=[1,0,1],linewidth=0.5)
    plt.plot(np.arange(-3,0,1/20),(cent_TRF[:,1]-cent_TRF_sem[:,1]),color=[1,0,1],linewidth=0.5)
    plt.plot([-3,0],[0,0],'k:')
    ax.axes.get_yaxis().set_ticks([])
    ax.axes.get_xaxis().set_ticks([])
    plt.ylim([-1*rangemax,rangemax])
    plt.title('UV')
    ax = plt.subplot(2,5,3)
    plt.plot(np.arange(-3,0,1/20),inner_TRF,color=[0.2,0.2,0.2])
    plt.plot(np.arange(-3,0,1/20),(inner_TRF+inner_TRF_sem),color=[0.2,0.2,0.2],linewidth=0.5)
    plt.plot(np.arange(-3,0,1/20),(inner_TRF-inner_TRF_sem),color=[0.2,0.2,0.2],linewidth=0.5)
    plt.plot([-3,0],[0,0],'k:')
    ax.axes.get_yaxis().set_ticks([])
    ax.axes.get_xaxis().set_ticks([])
    plt.ylim([-1*rangemax,rangemax])
    plt.title('Blue - UV')

    # SRFs - normalized to max value version
    # define max and min values
    maxval = np.max([np.max(blue_SRF),np.max(uv_SRF)])
    minval = np.min([np.min(blue_SRF),np.min(uv_SRF)])
    # find largest value
    rangemax = np.max([maxval,np.abs(minval)])
    # plotting
    x = plt.subplot(2,5,4);
    plt.imshow(blue_SRF/rangemax, origin='lower', cmap='PiYG', clim=[-1,1]);
    x.axes.get_xaxis().set_ticks([0,40,80])
    x.axes.get_yaxis().set_ticks([0,40,80])
    plt.title('Blue')
    x = plt.subplot(2,5,5);
    plt.imshow(uv_SRF/rangemax, origin='lower', cmap='PiYG', clim=[-1,1]);
    x.axes.get_xaxis().set_ticks([0,40,80])
    x.axes.get_yaxis().set_ticks([])
    plt.title('UV')
    plt.ylabel('Mean Spatial RF',fontsize='large', fontweight='bold')

    # spatial slices
    # define angles of slices to take through the mean SRF
    rotangles = np.arange(0,360,10)
    # container for slice data
    bslices = np.zeros((len(rotangles),blue_SRF.shape[1]))
    # spin SRF and grab slice from same index each time
    for n in range(0,len(rotangles)):
        bslices[n,:] = ndimage.rotate(blue_SRF,rotangles[n],reshape=False)[40,:]
    # repeat for UV
    uvslices = np.zeros((len(rotangles),uv_SRF.shape[1]))
    for n in range(0,len(rotangles)):
        uvslices[n,:] = ndimage.rotate(uv_SRF,rotangles[n],reshape=False)[40,:]
    # normalize slices to peak
    normval = np.max(np.abs([bslices.flatten(),uvslices.flatten()]))
    nbslices = bslices/normval
    nuvslices = uvslices/normval
    # plotting
    x = plt.subplot(2,5,9)
    plt.plot(np.mean(nbslices, axis=0), color=[0.1,0.2,0.8])
    plt.plot(np.mean(nbslices, axis=0)+np.std(nbslices, axis=0)/np.sqrt(n_cells), color=[0.1,0.2,0.8], linewidth=0.5)
    plt.plot(np.mean(nbslices, axis=0)-np.std(nbslices, axis=0)/np.sqrt(n_cells), color=[0.1,0.2,0.8], linewidth=0.5)
    plt.plot([0,80],[0,0],'k:')
    x.axes.get_xaxis().set_ticks([0,40,80])
    x.axes.get_yaxis().set_ticks([])
    plt.ylim(np.min([nbslices.flatten(),nuvslices.flatten()]),np.max([nbslices.flatten(),nuvslices.flatten()]))
    x = plt.subplot(2,5,10)
    plt.plot(np.mean(nuvslices, axis=0), color=[1,0,1])
    plt.plot(np.mean(nuvslices, axis=0)+np.std(nuvslices, axis=0)/np.sqrt(n_cells), color=[1,0,1], linewidth=0.5)
    plt.plot(np.mean(nuvslices, axis=0)-np.std(nuvslices, axis=0)/np.sqrt(n_cells), color=[1,0,1], linewidth=0.5)
    plt.plot([0,80],[0,0],'k:')
    x.axes.get_xaxis().set_ticks([0,40,80])
    x.axes.get_yaxis().set_ticks([])
    plt.ylabel('Norm. Radial RF',fontsize='large', fontweight='bold')
    plt.ylim(np.min([nbslices.flatten(),nuvslices.flatten()]),np.max([nbslices.flatten(),nuvslices.flatten()]))

    # surround TRFs
    ax = plt.subplot(2,5,6)
    rangemax = np.max([np.round(np.max(np.abs(surr_TRF)),2)+np.round(np.max(surr_TRF_sem),2)+0.02,0.1])
    plt.plot(np.arange(-3,0,1/20),surr_TRF[:,0],color=[0.1,0.2,0.8])
    plt.plot(np.arange(-3,0,1/20),(surr_TRF[:,0]+surr_TRF_sem[:,0]), color=[0.1,0.2,0.8],linewidth=0.5)
    plt.plot(np.arange(-3,0,1/20),(surr_TRF[:,0]-surr_TRF_sem[:,0]),color=[0.1,0.2,0.8],linewidth=0.5)
    plt.plot([-3,0],[0,0],'k:')
    plt.ylim([-1*rangemax,rangemax])
    ax.axes.get_xaxis().set_ticks([-3,-2,-1,0])
    plt.ylabel('Surrounds')
    ax = plt.subplot(2,5,7)
    plt.plot(np.arange(-3,0,1/20),surr_TRF[:,1],color=[1,0,1])
    plt.plot(np.arange(-3,0,1/20),(surr_TRF[:,1]+surr_TRF_sem[:,1]),color=[1,0,1],linewidth=0.5)
    plt.plot(np.arange(-3,0,1/20),(surr_TRF[:,1]-surr_TRF_sem[:,1]),color=[1,0,1],linewidth=0.5)
    plt.plot([-3,0],[0,0],'k:')
    ax.axes.get_xaxis().set_ticks([-3,-2,-1,0])
    ax.axes.get_yaxis().set_ticks([])
    plt.ylim([-1*rangemax,rangemax])
    plt.title('Mean Temporal Filters (units = z-score)',fontsize='large', fontweight='bold')
    ax = plt.subplot(2,5,8)
    plt.plot(np.arange(-3,0,1/20),surr_inner_TRF,color=[0.2,0.2,0.2])
    plt.plot(np.arange(-3,0,1/20),(surr_inner_TRF+surr_inner_TRF_sem),color=[0.2,0.2,0.2],linewidth=0.5)
    plt.plot(np.arange(-3,0,1/20),(surr_inner_TRF-surr_inner_TRF_sem),color=[0.2,0.2,0.2],linewidth=0.5)
    plt.plot([-3,0],[0,0],'k:')
    ax.axes.get_xaxis().set_ticks([-3,-2,-1,0])
    ax.axes.get_yaxis().set_ticks([])
    plt.ylim([-1*rangemax,rangemax])

    # save as PDF and close drawn figure
    fname = '/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/label summaries/individual cell types/'+label+'.pdf'
    sumfig.savefig(fname, format='pdf', orientation='landscape')
    plt.close()

#%% remove rows containing zeros from flattened mean response matrix and save it. the removed rows correspond to cell types for which the mean response was missing some part of the data
culled_small_flat = mean_small_flat_resp[np.unique(np.nonzero(mean_small_flat_resp)[0]),:]
included_labels = unique_types[np.unique(np.nonzero(mean_small_flat_resp)[0])]
np.save('/Volumes/TBD/Bruker/STRFs/compiled/culled_flattened_small_label_means.npy', culled_small_flat)

#%% mean TRF (blue,UV) by cell type
thresh_n = 0

# pull list of unique cell type labels
utype_indices = np.unique(scraped_log[1:,14]) != ''
unique_types = np.unique(scraped_log[1:,14])[utype_indices]

# save copy of mean TRFs for Alex's modeling - axes are t, c, type
all_mean_TRFs = np.zeros((60,2,len(unique_types)))

filtfig = plt.figure(figsize=(20, 20))
pltcnt=0
# plt.suptitle('Mean Center Filters for N>=3 Cell Types')
for ind in range(0,len(unique_types)):
    # define cell type to summarize
    label = unique_types[ind]
    # grab responses at indices where label matches cell type column
    label_STRFs = all_centered_STRFs[:,:,:,:,scraped_log[:,14] == label]
    n_cells = label_STRFs.shape[4]
    if n_cells > thresh_n:
        # defining TRF based on argmax of centered core (peak responding pixel in STRF center)
        label_cores = all_centered_cores[:,:,:,scraped_log[:,14] == label]
        label_TRFs = np.zeros(label_STRFs.shape[2:5])
        # grab the larger weight core that was actually used for centering, and find max
        for cell in range(0,n_cells):
            if np.abs(np.nansum(label_cores[:,:,0,cell],axis=(0,1))) > np.abs(np.nansum(label_cores[:,:,1,cell],axis=(0,1))):
                current_core = label_cores[:,:,0,cell].flatten()
                SRF_center_ind = np.argmax(np.abs(current_core))
                # take full timecourse for the max responding pixel
                label_TRFs[:,0,cell] = label_STRFs.reshape(-1,60,2,n_cells)[SRF_center_ind,:,0,cell]
                label_TRFs[:,1,cell] = label_STRFs.reshape(-1,60,2,n_cells)[SRF_center_ind,:,1,cell]
            else:
                current_core = label_cores[:,:,1,cell].flatten()
                SRF_center_ind = np.argmax(np.abs(current_core))
                label_TRFs[:,0,cell] = label_STRFs.reshape(-1,60,2,n_cells)[SRF_center_ind,:,0,cell]
                label_TRFs[:,1,cell] = label_STRFs.reshape(-1,60,2,n_cells)[SRF_center_ind,:,1,cell]
        # write to output container
        label_TRFs = np.nanmean(label_TRFs,axis=2)
        all_mean_TRFs[:,:,ind] = label_TRFs
        # plot
        rangemax = np.max([np.round(np.max(np.abs(label_TRFs)))+0.4,1]) #uses sliding cap w/ 1 as minimum
        plt.subplot(12,8,pltcnt+1)
        plt.plot(label_TRFs[30:,0],color=[0.1,0.2,0.8],linewidth=2)
        plt.plot(label_TRFs[30:,1],color=[1,0,1],linewidth=2)
        plt.plot([0,30],[0,0],'k:')
        plt.text(2,0.7,label,fontsize='large')
        plt.ylim([-1*rangemax,rangemax])
        pltcnt = pltcnt+1

# save means by cell type and cell type list
np.save('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/label summaries/all_mean_TRFs.npy', all_mean_TRFs)
np.save('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/label summaries/labels.npy', unique_types)

# save as PDF
fname = '/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/label summaries/all_mean_filters.pdf'
# fname = '/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/label summaries/n' + str(thresh_n) + '_mean_filters.pdf'
filtfig.savefig(fname, format='pdf', orientation='portrait')


#%% Cross-type summary statistics: flicker modulation depth, orientation tuning index, TRF FW@HM, TRF num lobes, TRF last lobe sign, SRF center size, ...

# load pre-calculated metrics
all_blue_resp = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all_blue_rois.npy')
all_uv_resp = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all_uv_rois.npy')
orient_selects = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all_orientation_selectivity_index.npy')
pref_orients = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all_pref_orientations.npy')
center_areas = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all_SRF_center_strict_areas.npy')
ellipse_areas = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all_ellipse_model_areas.npy')
long_axes = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all_long_axes.npy')
short_axes = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all_short_axes.npy')
mean_peak_SRFs = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all_mean_peak_SRFs.npy')
ds_index = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all_ds_index.npy')

all_flat_resp = np.zeros((scraped_log.shape[0], len(np.ravel(all_centered_STRFs[:,:,:,:,0])) + len(np.ravel(all_flick_resp[:,:,0]))), dtype='float')
all_small_flat_resp = np.zeros((scraped_log.shape[0], len(np.ravel(all_centered_STRFs[20:60,20:60,:,:,0])) + len(np.ravel(all_flick_resp[:,:,0]))), dtype='float')
all_short_small_flat_resp = np.zeros((scraped_log.shape[0], len(np.ravel(all_centered_STRFs[25:55,25:55,40:,:,0])) + len(np.ravel(all_flick_resp[:,:,0]))), dtype='float')
rot_flat_resp = np.zeros((scraped_log.shape[0], len(np.ravel(all_rotated_STRFs[:,:,:,:,0])) + len(np.ravel(all_flick_resp[:,:,0]))), dtype='float')

# de novo calculations of summary metrics
psd1D = np.zeros((39,2,scraped_log.shape[0]))
psd1D_pow = np.zeros((2,scraped_log.shape[0]))
pref_sfs = np.zeros(scraped_log.shape[0])
psdt = np.zeros((30,2,scraped_log.shape[0]))
psdt_pow = np.zeros((2,scraped_log.shape[0]))
psdt_surr = np.zeros((30,2,scraped_log.shape[0]))
psdt_surr_pow = np.zeros((2,scraped_log.shape[0]))
psdt_inner = np.zeros((30,scraped_log.shape[0]))
psdt_inner_pow = np.zeros((scraped_log.shape[0]))
pref_tfs = np.zeros(scraped_log.shape[0])
pref_tfs_surr = np.zeros(scraped_log.shape[0])
after_zc_auc = np.zeros((2,scraped_log.shape[0]))
after_zc_mean = np.zeros((2,scraped_log.shape[0]))
before_zc_auc = np.zeros((2,scraped_log.shape[0]))
surr_after_zc_auc = np.zeros((2,scraped_log.shape[0]))
surr_after_zc_mean = np.zeros((2,scraped_log.shape[0]))
surr_before_zc_auc = np.zeros((2,scraped_log.shape[0]))
spectral_sums = np.full((2,scraped_log.shape[0]), np.nan, dtype='float')
uvpi_center = np.full((scraped_log.shape[0]), np.NaN, dtype='float')
uvpi_surr = np.full((scraped_log.shape[0]), np.NaN, dtype='float')
flick_mod_depths = np.full((4,scraped_log.shape[0]), np.NaN, dtype='float')
flick_DCs = np.full((4,scraped_log.shape[0]), np.NaN, dtype='float')
all_cent_TRFs = np.full((all_centered_STRFs.shape[2],2,scraped_log.shape[0]), np.nan, dtype='float')
all_surr_TRFs = np.full((all_centered_STRFs.shape[2],2,scraped_log.shape[0]), np.nan, dtype='float')
all_cell_SRFs = np.zeros(all_centered_STRFs.shape[0:2]+(2,)+(scraped_log.shape[0],))
all_uncetered_cell_SRFs = np.zeros(all_blue_resp.shape[0:2]+(2,)+(scraped_log.shape[0],))

for cell_ind in range(0,scraped_log.shape[0]):
# for cell_ind in range(8,9):
    # grab responses at cell_ind index
    cell_STRFs = all_centered_STRFs[:,:,:,:,cell_ind]
    buc_STRF = all_blue_resp[:,:,:,cell_ind]
    uvuc_STRF = all_uv_resp[:,:,:,cell_ind]
    # cell_STRFs = all_rotated_STRFs[:,:,:,:,cell_ind]
    # defining TRF based on argmax of centered core (peak responding pixel in STRF center)
    cell_core = all_centered_cores[:,:,:,cell_ind]
    # cell_core = all_rotated_cores[:,:,:,cell_ind]
    # cell_surround_masks = all_rotated_surrounds[:,:,:,cell_ind]
    cell_surround_masks = all_centered_surrounds[:,:,:,cell_ind]
    cell_TRFs = np.zeros(cell_STRFs.shape[2:4])
    cell_surr_TRFs = np.zeros(cell_STRFs.shape[2:4])
    cell_inner_TRF = np.zeros(cell_TRFs.shape[0])
    cell_SRFs = np.zeros(cell_STRFs.shape[0:2]+(2,))
    cell_uc_SRFs = np.zeros(all_blue_resp.shape[0:2]+(2,))

    # grab the larger weight core that was actually used for centering, and find max
    if np.abs(np.nansum(cell_core[:,:,0],axis=(0,1))) > np.abs(np.nansum(cell_core[:,:,1],axis=(0,1))):
        current_core = cell_core[:,:,0].flatten()
        SRF_center_ind = np.argmax(np.abs(current_core))
        # take full timecourse for the max responding pixel
        cell_TRFs[:,0] = cell_STRFs.reshape(-1,60,2)[SRF_center_ind,:,0]
        cell_TRFs[:,1] = cell_STRFs.reshape(-1,60,2)[SRF_center_ind,:,1]
    else:
        current_core = cell_core[:,:,1].flatten()
        SRF_center_ind = np.argmax(np.abs(current_core))
        cell_TRFs[:,0] = cell_STRFs.reshape(-1,60,2)[SRF_center_ind,:,0]
        cell_TRFs[:,1] = cell_STRFs.reshape(-1,60,2)[SRF_center_ind,:,1]

    # use the surround mask to grab mean STRF activity over time in the region defined by the mask (for each color and cell)
    cell_surr_TRFs[:,0] = np.nanmean(cell_STRFs[np.abs(cell_surround_masks[:,:,0]) > 0,:,0],axis=0)
    cell_surr_TRFs[:,1] = np.nanmean(cell_STRFs[np.abs(cell_surround_masks[:,:,1]) > 0,:,1],axis=0)

    # inner PR contribution to center and surround TRFs is defined as UV-blue for each spatial segment
    cell_inner_TRF = cell_TRFs[:,1] - cell_TRFs[:,0]

    # save the center and surround TRFs to the all cells container array
    all_cent_TRFs[:,:,cell_ind] = cell_TRFs
    all_surr_TRFs[:,:,cell_ind] = cell_surr_TRFs

    # define flicker responses for cell
    cell_flicks = all_flick_resp[:,:,cell_ind]

    # find latest zero crossing of center for each TRF
    # find TRF index with largest value in each cell
    if not np.isnan(np.mean(cell_TRFs[:,0],axis=0)):
        # center
        peakind = np.argmax(np.abs(cell_TRFs[50:,0]))+50
        # find last zero crossings before and after peakind
        pre_diff = np.diff(np.sign(cell_TRFs[:peakind,0]))
        pre_ind = np.max(np.where(pre_diff)) + 1
        try:
            post_diff = np.diff(np.sign(cell_TRFs[peakind:,0]))
            post_ind = peakind + np.min(np.where(post_diff))
        except:
            post_ind = 59
        # take mean SRF for period defined by these indices
        cell_SRFs[:,:,0] = np.nanmean(cell_STRFs[:,:,pre_ind:post_ind,0],axis=2)
        cell_uc_SRFs[:,:,0] = np.nanmean(buc_STRF[:,:,pre_ind:post_ind],axis=2)
        # find AUC defined by pre-post period, and AUC before this period
        after_zc_auc[0,cell_ind] = np.nansum(cell_TRFs[pre_ind:post_ind,0])
        after_zc_mean[0,cell_ind] = np.nanmax(np.abs(cell_TRFs[pre_ind:post_ind,0]))*np.sign(cell_TRFs[peakind,0])
        before_zc_auc[0,cell_ind] = np.nansum(cell_TRFs[20:pre_ind,0]) # "before" looks 2s into the past

        # surround
        peakind = np.argmax(np.abs(cell_surr_TRFs[50:,0]))+50
        # find last zero crossings before and after peakind
        pre_diff = np.diff(np.sign(cell_surr_TRFs[:peakind,0]))
        pre_ind = np.max(np.where(pre_diff)) + 1
        try:
            post_diff = np.diff(np.sign(cell_surr_TRFs[peakind:,0]))
            post_ind = peakind + np.min(np.where(post_diff))
        except:
            post_ind = 59
        # find AUC defined by pre-post period, and AUC before this period
        surr_after_zc_auc[0,cell_ind] = np.nansum(cell_surr_TRFs[pre_ind:post_ind,0])
        surr_after_zc_mean[0,cell_ind] = np.nanmax(np.abs(cell_surr_TRFs[pre_ind:post_ind,0]))*np.sign(cell_surr_TRFs[peakind,0])
        surr_before_zc_auc[0,cell_ind] = np.nansum(cell_surr_TRFs[20:pre_ind,0])

    # perform same operation for uv SRF
    if not np.isnan(np.mean(cell_TRFs[:,1],axis=0)):
        # center
        peakind = np.argmax(np.abs(cell_TRFs[50:,1]))+50
        # find last zero crossings before and after peakind
        pre_diff = np.diff(np.sign(cell_TRFs[:peakind,1]))
        pre_ind = np.max(np.where(pre_diff)) + 1
        try:
            post_diff = np.diff(np.sign(cell_TRFs[peakind:,1]))
            post_ind = peakind + np.min(np.where(post_diff))
        except:
            post_ind = 59
        # take SRF for period defined by these indices
        cell_SRFs[:,:,1] = np.nanmean(cell_STRFs[:,:,pre_ind:post_ind,1],axis=2)
        cell_uc_SRFs[:,:,1] = np.nanmean(uvuc_STRF[:,:,pre_ind:post_ind],axis=2)
        # find AUC defined by pre-post period, and AUC before this period
        after_zc_auc[1,cell_ind] = np.nansum(cell_TRFs[pre_ind:post_ind,1])
        after_zc_mean[1,cell_ind] = np.nanmax(np.abs(cell_TRFs[pre_ind:post_ind,1]))*np.sign(cell_TRFs[peakind,1])
        before_zc_auc[1,cell_ind] = np.nansum(cell_TRFs[20:pre_ind,1])

        # surround
        peakind = np.argmax(np.abs(cell_surr_TRFs[50:,1]))+50
        # find last zero crossings before and after peakind
        pre_diff = np.diff(np.sign(cell_surr_TRFs[:peakind,0]))
        pre_ind = np.max(np.where(pre_diff)) + 1
        try:
            post_diff = np.diff(np.sign(cell_surr_TRFs[peakind:,1]))
            post_ind = peakind + np.min(np.where(post_diff))
        except:
            post_ind = 59
        # find AUC defined by pre-post period, and AUC before this period
        surr_after_zc_auc[1,cell_ind] = np.nansum(cell_surr_TRFs[pre_ind:post_ind,1])
        surr_after_zc_mean[1,cell_ind] = np.nanmax(np.abs(cell_surr_TRFs[pre_ind:post_ind,1]))*np.sign(cell_surr_TRFs[peakind,1])
        surr_before_zc_auc[1,cell_ind] = np.nansum(cell_surr_TRFs[20:pre_ind,1])

    # save SRFs to array
    all_cell_SRFs[:,:,:,cell_ind] = cell_SRFs
    all_uncetered_cell_SRFs[:,:,:,cell_ind] = cell_uc_SRFs

    # calculate UV preference indices for center and surround based on the difference between uv and blue aucs. If the signs of the metrics are different, a perfect PI of 1 or -1 will be assigned based on whether the uv or blue response amplitude is larger, respectively
    if np.isnan(np.mean(cell_TRFs[:,0],axis=0)) or np.isnan(np.mean(cell_TRFs[:,1],axis=0)):
        pass
    else:
        # determine sign of uv response and sign difference - if diff = -1, want to add blue to UV (this essentially creates a perfect 1 or -1 in every case where blue and UV give different center signs, which is rare)
        uv_sign = np.sign(after_zc_mean[1,cell_ind])
        sign_diff = uv_sign*np.sign(after_zc_mean[0,cell_ind])
        # multiply each auc by the UV sign - ensures that all UV responses are positive and all blue responses are negative
        uvpi_center[cell_ind] = (uv_sign*after_zc_mean[1,cell_ind] - sign_diff*np.abs(after_zc_mean[0,cell_ind])) / (np.abs(after_zc_mean[1,cell_ind]) + np.abs(after_zc_mean[0,cell_ind]))
        # this algorithm works for all circumstances EXCEPT when BOTH the blue response is larger than the UV response AND they have different signs. without this manual adjustment, this situation will yield uvpi = 1.
        if sign_diff == -1 and (np.abs(after_zc_mean[0,cell_ind]) > np.abs(after_zc_mean[1,cell_ind])):
            uvpi_center[cell_ind] = -1
        # repeat for surround
        uv_sign = np.sign(surr_after_zc_mean[1,cell_ind])
        sign_diff = uv_sign*np.sign(surr_after_zc_mean[0,cell_ind])
        uvpi_surr[cell_ind] = (uv_sign*surr_after_zc_mean[1,cell_ind] - sign_diff*np.abs(surr_after_zc_mean[0,cell_ind])) / (np.abs(surr_after_zc_mean[1,cell_ind]) + np.abs(surr_after_zc_mean[0,cell_ind]))
        if sign_diff == -1 and (np.abs(surr_after_zc_mean[0,cell_ind]) > np.abs(surr_after_zc_mean[1,cell_ind])):
            uvpi_surr[cell_ind] = -1

    # spatial and temporal frequency power spectra
    for c in range(0,2):
        # calculate 1-D spatial power spectrum using Max's method
        # fft_2d = np.abs(np.fft.fftshift(np.fft.fft2(image_array[:, :512])))**2
        fft_2d = np.abs(np.fft.fftshift(np.fft.fft2(cell_SRFs[:,:,c])))**2
        ndim = fft_2d.shape[0]
        pixels_per_degree = 1
        # Circular sum to collapse into 1D power spectrum
        # Ref: https://medium.com/tangibit-studios/2d-spectrum-characterization-e288f255cc59
        h = fft_2d.shape[0]
        w = fft_2d.shape[1]
        wc = w//2
        hc = h//2
        # create an array of integer radial distances from the center
        Y, X = np.ogrid[0:h, 0:w]
        r = np.hypot(X - wc, Y - hc).astype(int)
        # SUM all psd2D pixels with label 'r' for 0<=r<=wc
        # NOTE: this will miss power contributions in 'corners' r>wc
        cell_psd = ndimage.sum(fft_2d, r, index=np.arange(0, wc))
        freq = np.fft.fftfreq(ndim, d=pixels_per_degree)[:ndim//2]
        # normalize by sum over frequencies
        psd1D_pow[c,cell_ind] = np.nansum(cell_psd[1:])
        psd1D[:,c,cell_ind] = cell_psd[1:]/np.nansum(cell_psd[1:])

        # calculate power spectrum on temporal kernel of RF center
        tfft = np.abs(np.fft.fft(cell_TRFs[:,c]))
        update_period = 1/20
        tndim = tfft.shape[0]
        tfreq = np.fft.fftfreq(tndim, d=update_period)[:tndim//2]
        # normalize by sum over frequencies
        psdt_pow[c,cell_ind] = np.nansum(tfft[:tndim//2])
        psdt[:,c,cell_ind] = tfft[:tndim//2]/np.nansum(tfft[:tndim//2])
        # take peak of center as preferred TF/SF. treats blue and UV as independent observations and pulls a single largest power. In essence, asks for the preference of the more selective color
        if not np.isnan(np.amax(psd1D[:,:,cell_ind])):
            pref_sfs[cell_ind] = freq[np.where(psd1D[:,:,cell_ind]==np.amax(psd1D[:,:,cell_ind]))[0][0]]
            pref_tfs[cell_ind] = tfreq[np.where(psdt[:,:,cell_ind]==np.amax(psdt[:,:,cell_ind]))[0][0]]

        # calculate power spectrum on temporal kernel of RF surround
        tfft = np.abs(np.fft.fft(cell_surr_TRFs[:,c]))
        update_period = 1/20
        tndim = tfft.shape[0]
        tfreq = np.fft.fftfreq(tndim, d=update_period)[:tndim//2]
        # normalize by sum over frequencies
        psdt_surr_pow[c,cell_ind] = np.nansum(tfft[:tndim//2])
        psdt_surr[:,c,cell_ind] = tfft[:tndim//2]/np.nansum(tfft[:tndim//2])
        if not np.isnan(np.amax(psd1D[:,:,cell_ind])):
            pref_tfs_surr[cell_ind] = tfreq[np.where(psdt_surr[:,:,cell_ind]==np.amax(psdt_surr[:,:,cell_ind]))[0][0]]

    # calculate power spectrum on temporal kernel of inner photoreceptor contribution
    tfft = np.abs(np.fft.fft(cell_inner_TRF))
    update_period = 1/20
    tndim = tfft.shape[0]
    tfreq = np.fft.fftfreq(tndim, d=update_period)[:tndim//2]
    # normalize by sum over frequencies
    psdt_inner_pow[cell_ind] = np.nansum(tfft[:tndim//2])
    psdt_inner[:,cell_ind] = tfft[:tndim//2]/np.nansum(tfft[:tndim//2])

    # flicker modulation depth and DC signal by Frequency
    if not np.isnan(cell_flicks[0,0]):
        # each cycle goes 0.5 -> 1 -> 0.5 -> 0 -> 0.5
        # for 0.1 Hz (1 cycle), simply take min and max over 2.1-12.1 sec [42:242]
        flick_mod_depths[0,cell_ind] = np.amax(cell_flicks[0,10:60]) - np.amin(cell_flicks[0,10:60])
        flick_DCs[0,cell_ind] = np.amin(cell_flicks[0,10:60])
        # for all other frequencies, calculate mod depth on each cycle and take the average
        for flick_ind in range(1,4):
            # each index is 0.2 sec, so want to take 20/freq indices for each cycle
            cyc_period = int(5/flick_freqs[flick_ind]) #units = samples
            num_cyc = int(10*5/cyc_period)
            freq_mds = np.zeros(num_cyc)
            freq_DCs = np.zeros(num_cyc)
            for cycle in range(0,num_cyc):
                freq_mds[cycle] = np.amax(cell_flicks[flick_ind,10+(cycle*cyc_period):10+((cycle+1)*cyc_period)]) - np.amin(cell_flicks[flick_ind,10+(cycle*cyc_period):10+((cycle+1)*cyc_period)])
                freq_DCs[cycle] = np.amin(cell_flicks[flick_ind,10+(cycle*cyc_period):10+((cycle+1)*cyc_period)])
            flick_mod_depths[flick_ind,cell_ind] = np.mean(freq_mds)
            # the DC signal is the average of the cycle minima
            flick_DCs[flick_ind,cell_ind] = np.mean(freq_DCs)

    # flatten each response into a single vector (for PCA and clustering)
    # PCA and clustering do not accept NaNs, so compile a matrix that skips cells with missing data, and delete zero rows later
    if not np.isnan(cell_flicks[0,0]):
        if not np.any(np.isnan(cell_STRFs)):
            # the inverse of this procedure is np.reshape(all_flat_resp[:768000], all_centered_STRFs.shape[0:4])
            # flat_STRF = np.ravel(cell_STRFs)
            rot_STRF = np.ravel(cell_STRFs)
            # the inverse of this procedure is np.reshape(all_flat_resp[:192000], all_centered_STRFs[20:60,20:60,:,:,:].shape[0:4])
            small_flat_STRF = np.ravel(cell_STRFs[20:60,20:60,:,:])
            # the inverse of this procedure is np.reshape(all_flat_resp[:36000], all_centered_STRFs[25:55,25:55,40:,:,:].shape[0:4])
            short_small_flat_STRF = np.ravel(cell_STRFs[25:55,25:55,40:,:])
            # the inverse of this procedure is np.reshape(all_flat_resp[512000:], all_flick_resp.shape[0:2]) (or np.reshape(all_flat_resp[128000:], all_flick_resp.shape[0:2]) for the small flattened response matrix)
            flat_flicks = np.ravel(cell_flicks)
            # all_flat_resp[cell_ind,:] = np.append(flat_STRF,flat_flicks)
            all_small_flat_resp[cell_ind,:] = np.append(small_flat_STRF,flat_flicks)
            all_short_small_flat_resp[cell_ind,:] = np.append(short_small_flat_STRF,flat_flicks)
            rot_flat_resp[cell_ind,:] = np.append(rot_STRF,flat_flicks)

#%%
culled_short_flat = all_short_small_flat_resp[np.unique(np.nonzero(all_short_small_flat_resp)[0]),:]
np.save('/Volumes/TBD/Bruker/STRFs/compiled/culled_flattened_short_small_responses.npy', culled_short_flat)

#%%
# remove zero rows from flattened response matrix and save it for PCA
culled_flat_resp = all_flat_resp[np.unique(np.nonzero(all_flat_resp)[0]),:]
culled_small_flat = all_small_flat_resp[np.unique(np.nonzero(all_small_flat_resp)[0]),:]
culled_flat_rot = rot_flat_resp[np.unique(np.nonzero(rot_flat_resp)[0]),:]
np.save('/Volumes/TBD/Bruker/STRFs/compiled/culled_flattened_responses.npy', culled_flat_resp)
np.save('/Volumes/TBD/Bruker/STRFs/compiled/culled_flattened_rotated_responses.npy', culled_flat_rot)
np.save('/Volumes/TBD/Bruker/STRFs/compiled/culled_flattened_small_responses.npy', culled_small_flat)

#%% save new preference indices
np.save('/Volumes/TBD/Bruker/STRFs/compiled/all_spectral_pref_index_CENTER.npy', uvpi_center)
np.save('/Volumes/TBD/Bruker/STRFs/compiled/all_spectral_pref_index_SURROUND.npy', uvpi_surr)
np.save('/Volumes/TBD/Bruker/STRFs/compiled/all_pref_tempfreq_CENTER.npy', pref_tfs)
np.save('/Volumes/TBD/Bruker/STRFs/compiled/all_pref_tempfreq_SURROUND.npy', pref_tfs_surr)
np.save('/Volumes/TBD/Bruker/STRFs/compiled/all_pref_spacefreq.npy', pref_sfs)

#%% save temporal data and AUC summaries
np.save('/Volumes/TBD/Bruker/STRFs/compiled/all_center_TRFs.npy', all_cent_TRFs)
np.save('/Volumes/TBD/Bruker/STRFs/compiled/all_surround_TRFs.npy', all_surr_TRFs)
np.save('/Volumes/TBD/Bruker/STRFs/compiled/all_last_lobe_peak_z.npy', after_zc_mean)
np.save('/Volumes/TBD/Bruker/STRFs/compiled/all_last_lobe_peak_z_SURROUND.npy', surr_after_zc_mean)

#%% save summary metrics for feature vector; full and missing-data culled
np.save('/Volumes/TBD/Bruker/STRFs/compiled/all_spatial_spectra.npy', psd1D)
np.save('/Volumes/TBD/Bruker/STRFs/compiled/all_temporal_spectra.npy', psdt)
np.save('/Volumes/TBD/Bruker/STRFs/compiled/all_surround_temporal_spectra.npy', psdt_surr)
np.save('/Volumes/TBD/Bruker/STRFs/compiled/all_spatial_power.npy', psd1D_pow)
np.save('/Volumes/TBD/Bruker/STRFs/compiled/all_temporal_power.npy', psdt_pow)
np.save('/Volumes/TBD/Bruker/STRFs/compiled/all_surround_temporal_power.npy', psdt_surr_pow)
np.save('/Volumes/TBD/Bruker/STRFs/compiled/all_before_last_lobe_AUC.npy', before_zc_auc)
np.save('/Volumes/TBD/Bruker/STRFs/compiled/all_last_lobe_AUC.npy', after_zc_auc)
np.save('/Volumes/TBD/Bruker/STRFs/compiled/all_surround_before_last_lobe_AUC.npy', surr_before_zc_auc)
np.save('/Volumes/TBD/Bruker/STRFs/compiled/all_surround_last_lobe_AUC.npy', surr_after_zc_auc)
np.save('/Volumes/TBD/Bruker/STRFs/compiled/all_flicker_MDs.npy', flick_mod_depths)
np.save('/Volumes/TBD/Bruker/STRFs/compiled/all_flicker_DCs.npy', flick_DCs)

np.save('/Volumes/TBD/Bruker/STRFs/compiled/culled_spatial_spectra.npy', psd1D[:,:,np.unique(np.nonzero(all_flat_resp)[0])])
np.save('/Volumes/TBD/Bruker/STRFs/compiled/culled_temporal_spectra.npy', psdt[:,:,np.unique(np.nonzero(all_flat_resp)[0])])
np.save('/Volumes/TBD/Bruker/STRFs/compiled/culled_surround_temporal_spectra.npy', psdt_surr[:,:,np.unique(np.nonzero(all_flat_resp)[0])])
np.save('/Volumes/TBD/Bruker/STRFs/compiled/culled_spatial_power.npy', psd1D_pow[:,np.unique(np.nonzero(all_flat_resp)[0])])
np.save('/Volumes/TBD/Bruker/STRFs/compiled/culled_temporal_power.npy', psdt_pow[:,np.unique(np.nonzero(all_flat_resp)[0])])
np.save('/Volumes/TBD/Bruker/STRFs/compiled/culled_surround_temporal_power.npy', psdt_surr_pow[:,np.unique(np.nonzero(all_flat_resp)[0])])
np.save('/Volumes/TBD/Bruker/STRFs/compiled/culled_before_last_lobe_AUC.npy', before_zc_auc[:,np.unique(np.nonzero(all_flat_resp)[0])])
np.save('/Volumes/TBD/Bruker/STRFs/compiled/culled_last_lobe_AUC.npy', after_zc_auc[:,np.unique(np.nonzero(all_flat_resp)[0])])
np.save('/Volumes/TBD/Bruker/STRFs/compiled/culled_surround_before_last_lobe_AUC.npy', surr_before_zc_auc[:,np.unique(np.nonzero(all_flat_resp)[0])])
np.save('/Volumes/TBD/Bruker/STRFs/compiled/culled_surround_last_lobe_AUC.npy', surr_after_zc_auc[:,np.unique(np.nonzero(all_flat_resp)[0])])
np.save('/Volumes/TBD/Bruker/STRFs/compiled/culled_flicker_MDs.npy', flick_mod_depths[:,np.unique(np.nonzero(all_flat_resp)[0])])
np.save('/Volumes/TBD/Bruker/STRFs/compiled/culled_flicker_DCs.npy', flick_DCs[:,np.unique(np.nonzero(all_flat_resp)[0])])

#%% cull and save spatial metrics

# spatial_metrics = []
# spatial_metrics = np.append([np.unique(np.nonzero(all_flat_resp)[0])], center_areas[np.unique(np.nonzero(all_flat_resp)[0])])
# spatial_metrics = np.append(spatial_metrics, long_axes[np.unique(np.nonzero(all_flat_resp)[0])])
# spatial_metrics = np.append(spatial_metrics, short_axes[np.unique(np.nonzero(all_flat_resp)[0])])
#
# spatial_metrics = np.reshape(spatial_metrics,(-1,len(np.unique(np.nonzero(all_flat_resp)[0]))))
#
# np.save('/Volumes/TimBigData/Bruker/STRFs/compiled/culled_spatial_metrics.npy', spatial_metrics)
spatial_metrics = []
spatial_metrics = np.append(orient_selects, center_areas)
spatial_metrics = np.append(spatial_metrics, ellipse_areas)
spatial_metrics = np.append(spatial_metrics, long_axes)
spatial_metrics = np.append(spatial_metrics, short_axes)

spatial_metrics = np.reshape(spatial_metrics,(-1,scraped_log.shape[0]))
np.save('/Volumes/TBD/Bruker/STRFs/compiled/all_spatial_metrics.npy', spatial_metrics)
# NEXT: calculate nonlinearity for each neuron; add to overall flicker summary figure (plot each cell's nonlinearity as a line in an imshow)
# 3: by cluster or cell type spatial and temporal numbers

#%% cull and save STRF and flicker data

np.save('/Volumes/TBD/Bruker/STRFs/compiled/culled_centered_STRFs.npy', all_centered_STRFs[:,:,:,:,np.unique(np.nonzero(all_flat_resp)[0])])
np.save('/Volumes/TBD/Bruker/STRFs/compiled/culled_rotated_STRFs.npy', all_rotated_STRFs[:,:,:,:,np.unique(np.nonzero(all_flat_resp)[0])])
np.save('/Volumes/TBD/Bruker/STRFs/compiled/culled_flick_responses.npy', all_flick_resp[:,:,np.unique(np.nonzero(all_flat_resp)[0])])

#%% open, cull, and save the shift table
all_shifts = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all_shifts.npy')
np.save('/Volumes/TBD/Bruker/STRFs/compiled/culled_shifts.npy', all_shifts[:,np.unique(np.nonzero(all_flat_resp)[0])])


#%% temporal summary figure
# to try: plot power spectrum as an image, sorted by peak location
# to try: temporal filters as an image, sorted by zero crossing time?
utype_indices = np.unique(scraped_log[:,14]) != ''
unique_types = np.unique(scraped_log[:,14])[utype_indices]

tempfig = plt.figure(figsize=(15, 9))
plt.suptitle('Temporal Summary - All Neurons',fontsize='xx-large', fontweight='bold')

plt.subplot(2,3,1,aspect=0.6)
plt.plot([-20,20],[0,0],'k--',linewidth=0.5)
plt.plot([0,0],[-30,30],'k--',linewidth=0.5)
plt.plot([-20,20],[30,-30],'k--',linewidth=0.5)
plt.xlim(-20,20)
plt.ylim(-30,30)
plt.ylabel('Last Lobe AUC')
plt.xlabel('Before Last Lobe AUC')
plt.text(-15,25,'ON', fontsize='large', fontweight='demibold')
plt.text(-7,-10,'Differentiators', fontsize='large', fontweight='demibold', rotation=-43)
plt.text(10,-27,'OFF', fontsize='large', fontweight='demibold')
plt.text(-7,-24,'Integrators', fontsize='large', fontweight='demibold')
plt.text(-7,22,'Integrators', fontsize='large', fontweight='demibold')
plt.text(-17,1,'Delayed', fontsize='large', fontweight='demibold')
plt.text(-19,-3,'Integrators', fontsize='large', fontweight='demibold')
plt.text(8,1,'Delayed', fontsize='large', fontweight='demibold')
plt.text(6,-3,'Integrators', fontsize='large', fontweight='demibold')

plt.subplot(2,3,2,aspect=0.6)
plt.xlim(-15,15)
plt.ylim(-20,20)
plt.plot([-20,20],[0,0],'k--',linewidth=0.5)
plt.plot([0,0],[-30,30],'k--',linewidth=0.5)
plt.plot([-20,20],[30,-30],'k--',linewidth=0.5)
plt.scatter(before_zc_auc[0,:],after_zc_auc[0,:],s=10,color=[0.1,0.1,0.8])
plt.scatter(before_zc_auc[1,:],after_zc_auc[1,:],s=10,color=[0.8,0.1,0.1])
plt.ylabel('Last Lobe AUC')
plt.xlabel('Before Last Lobe AUC')


# plt.subplot(2,3,3)
# plt.xlim(-0.5,10)
# plt.plot([-0.5,10],[0,0],'k--',linewidth=0.5)
# plt.scatter(pref_tfs,after_zc_auc[0,:]-before_zc_auc[0,:],s=10,color=[0.1,0.1,0.8])
# plt.scatter(pref_tfs,after_zc_auc[1,:]-before_zc_auc[1,:],s=10,color=[0.8,0.1,0.1])
# plt.ylabel('AUC difference (last lobe - before)')
# plt.xlabel('Preferred TF')

plt.subplot(2,3,4)
tf_hist = np.histogram(pref_tfs_surr,bins=30,range=(0,10))
tf_norm = tf_hist[0]/np.sum(tf_hist[0])
plt.plot(tf_hist[1][1:],tf_norm,color=[0.2,0.5,0.5]);
tf_hist = np.histogram(pref_tfs,bins=30,range=(0,10))
tf_norm = tf_hist[0]/np.sum(tf_hist[0])
plt.plot(tf_hist[1][1:],tf_norm,color=[0.8,0.5,0.2]);
plt.ylabel('Probability')
plt.xlabel('Preferred Temporal Freq (Hz)')

plt.subplot(2,3,5)
uvpi_hist = np.histogram(uvpi_surr,bins=30,range=(-1,1))
uvpi_norm = uvpi_hist[0]/np.sum(uvpi_hist[0])
plt.plot(uvpi_hist[1][1:],uvpi_norm,color=[0.2,0.5,0.5]);
uvpi_hist = np.histogram(uvpi_center,bins=30,range=(-1,1))
uvpi_norm = uvpi_hist[0]/np.sum(uvpi_hist[0])
plt.plot(uvpi_hist[1][1:],uvpi_norm,color=[0.8,0.5,0.2]);
plt.ylabel('Probability')
plt.xlabel('Spectral Preference Index')

# plt.subplot(2,3,5)
tf_vec = tfreq[1:]
norm_nt_ps = (1/tfreq[1:])/np.sum(1/tfreq[1:])
mean_temp_ps = np.nanmean(psdt,axis=(1,2))
# plt.loglog(tfreq,mean_temp_ps)
# # plt.loglog(tfreq,psdt[:,1,327])
# # plt.loglog(tfreq,psdt[:,1,327])
# plt.loglog(tf_vec,norm_nt_ps,'k-')
# plt.ylabel('Log(Norm. Power)')
# plt.xlabel('Temporal Frequency')

plt.subplot(2,3,3)
plt.loglog(tf_vec,norm_nt_ps,'k-')
mean_label_spectrum = np.zeros((psdt.shape[0],len(unique_types)))
for ind in range(0,len(unique_types)):
    current_label = unique_types[ind]
    label_spectra = psdt_surr[:,:,scraped_log[:,14] == current_label]
    mean_label_spectrum[:,ind] = np.nanmean(label_spectra,axis=(1,2))
    plt.loglog(tfreq,mean_label_spectrum[:,ind],color=[0.4,0.4,0.4],linewidth=0.5)
plt.loglog(tfreq,np.nanmean(psdt_surr,axis=(1,2)),linewidth=2,color=[0.2,0.5,0.5])
plt.ylim([0.003, 0.15])
plt.ylabel('Log(Norm. Power)')
plt.xlabel('Temporal Frequency')
plt.title('Temporal Spectra of RF Surrounds')

plt.subplot(2,3,6)
plt.loglog(tf_vec,norm_nt_ps,'k-')
mean_label_spectrum = np.zeros((psdt.shape[0],len(unique_types)))
# for ind in range(0,1):
for ind in range(0,len(unique_types)):
    current_label = unique_types[ind]
    label_spectra = psdt[:,:,scraped_log[:,14] == current_label]
    mean_label_spectrum[:,ind] = np.nanmean(label_spectra,axis=(1,2))
    plt.loglog(tfreq,mean_label_spectrum[:,ind],color=[0.4,0.4,0.4],linewidth=0.5)
plt.loglog(tfreq,mean_temp_ps,linewidth=2,color=[0.8,0.5,0.2])
plt.ylim([0.003, 0.15])
plt.ylabel('Log(Norm. Power)')
plt.xlabel('Temporal Frequency')
plt.title('Temporal Spectra of RF Centers')



# save it
fname = '/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all cell summaries/temporal_summary.pdf'
tempfig.savefig(fname, format='pdf', orientation='landscape')
#%% By cell type / advanced temporal summary for RF centers

# this value determines the threshold for N that will be required to include a cell type in summary plots (non-inclusive lower bound)
thresh_n = 2

utype_indices = np.unique(scraped_log[:,14]) != ''
unique_types = np.unique(scraped_log[:,14])[utype_indices]
classmarkers = np.zeros(len(unique_types))
signmarkers = np.zeros(len(unique_types))
Nbytype = np.zeros(len(unique_types))

mean_label_spectrum = np.zeros((psdt.shape[0],2,len(unique_types)))
uvpi_means = np.zeros(len(unique_types))

tempfig = plt.figure(figsize=(15, 15))
plt.suptitle('Temporal Summary for RF Centers, by Cell Type (N>=3)',fontsize='xx-large', fontweight='bold')
# build axes
plt.subplot(3,3,1,aspect=0.6)
plt.plot([-20,20],[0,0],'k--',linewidth=0.5)
plt.plot([0,0],[-30,30],'k--',linewidth=0.5)
plt.plot([-20,20],[20,-20],'k--',linewidth=0.5)
plt.xlim(-20,20)
plt.ylim(-30,30)
plt.ylabel('Last Lobe AUC')
plt.xlabel('Before Last Lobe AUC')
plt.text(-15,25,'ON', fontsize='large', fontweight='demibold')
plt.text(-8,-9,'Differentiators', fontsize='large', fontweight='demibold', rotation=-30)
plt.text(10,-27,'OFF', fontsize='large', fontweight='demibold')
plt.text(-7,-24,'Integrators', fontsize='large', fontweight='demibold')
plt.text(-7,22,'Integrators', fontsize='large', fontweight='demibold')
plt.text(-17,1,'Delayed', fontsize='large', fontweight='demibold')
plt.text(-19,-3,'Integrators', fontsize='large', fontweight='demibold')
plt.text(8,1,'Delayed', fontsize='large', fontweight='demibold')
plt.text(6,-3,'Integrators', fontsize='large', fontweight='demibold')

plt.subplot(3,3,2,aspect=0.6)
plt.plot([-20,20],[0,0],'k--',linewidth=0.5)
plt.plot([0,0],[-30,30],'k--',linewidth=0.5)
plt.plot([-20,20],[20,-20],'k--',linewidth=0.5)
plt.xlim(-8,8)
plt.ylim(-12,12)
plt.title('Blue')
plt.ylabel('Last Lobe AUC')
plt.xlabel('Before Last Lobe AUC')
plt.subplot(3,3,3,aspect=0.6)
plt.plot([-20,20],[0,0],'k--',linewidth=0.5)
plt.plot([0,0],[-30,30],'k--',linewidth=0.5)
plt.plot([-20,20],[20,-20],'k--',linewidth=0.5)
plt.xlim(-8,8)
plt.ylim(-12,12)
plt.title('UV')
plt.ylabel('Last Lobe AUC')
plt.xlabel('Before Last Lobe AUC')

plt.subplot(3,3,4)
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.plot([-20,20],[0,0],'k--',linewidth=0.5)
plt.plot([0,0],[-20,20],'k--',linewidth=0.5)
plt.ylabel('Center Spectral Index')
plt.xlabel('Surround Spectral Index')



for ind in range(0,len(unique_types)):
    # pull summary measurements for current cell type
    current_label = unique_types[ind]
    current_cells = scraped_log[scraped_log[:,14] == current_label,1]
    current_n = len(current_cells)
    Nbytype[ind] = current_n

    if current_n > thresh_n:
        # temporal spectrum
        label_spectra = psdt[:,:,scraped_log[:,14] == current_label]
        mean_label_spectrum[:,:,ind] = np.nanmean(label_spectra,axis=2)
        # preferred freq
        label_prefTFs = pref_tfs[scraped_log[:,14] == current_label]
        # TFmeans[ind,2] = np.nanmean(label_prefTFs)
        # AUCs
        label_BLLauc = before_zc_auc[:,scraped_log[:,14] == current_label]
        label_LLauc = after_zc_auc[:,scraped_log[:,14] == current_label]
        label_uvpi_center = uvpi_center[scraped_log[:,14] == current_label]
        uvpi_means[ind] = np.nanmedian(label_uvpi_center)
        label_uvpi_surr = uvpi_surr[scraped_log[:,14] == current_label]
        blue_sem = np.mean([np.std(label_BLLauc[0,:])/np.sqrt(current_n),np.std(label_LLauc[0,:])/np.sqrt(current_n)])
        uv_sem = np.mean([np.std(label_BLLauc[1,:])/np.sqrt(current_n),np.std(label_LLauc[1,:])/np.sqrt(current_n)])

        # define plot color based on cell class, and tag marker vector
        if np.logical_or(np.logical_or(current_label[0]=='P',current_label[0]=='D'),current_label[0]=='S'):
            plot_color=[0.8,0.2,0.8]
            classmarkers[ind] = 3
            # plt.subplot(339)
            # plt.loglog(tfreq,np.nanmean(label_spectra,axis=(1,2)),color=plot_color,linewidth=0.5)
        elif np.logical_or(current_label[0]=='C',current_label[0]=='Y'):
            plot_color=[0.2,0.8,0.2]
            classmarkers[ind] = 2
            # plt.subplot(338)
            # plt.loglog(tfreq,np.nanmean(label_spectra,axis=(1,2)),color=plot_color,linewidth=0.5)
        else:
            plot_color=[0,0,0]
            classmarkers[ind] = 1
            # plt.subplot(337)
            # plt.loglog(tfreq,np.nanmean(label_spectra,axis=(1,2)),color=plot_color,linewidth=0.5)
        # manual correction for LMa's and Olt
        if np.logical_or(current_label[0:1]=='LM',current_label[0:1]=='Ol'):
            plot_color=[0.8,0.2,0.8]
            classmarkers[ind] = 3
            # plt.subplot(339)
            # plt.loglog(tfreq,np.nanmean(label_spectra,axis=(1,2)),color=plot_color,linewidth=0.5)

        # plot spectra based on sign of last lobe AUC
        if np.sign(np.nanmedian(label_LLauc)) == 1:
            signmarkers[ind] = 1
            # Spectral selectivity index - AUC diff blue-UV
            plt.subplot(3,3,4)
            plt.scatter(np.nanmean(label_uvpi_surr), np.nanmean(label_uvpi_center), color=[0.8,0.1,0.1], s=2+400*current_n/61, alpha=0.9-uv_sem/4)
            # plt.subplot(337)
            # plt.loglog(tfreq,np.nanmean(label_spectra,axis=(1,2)),color=plot_color,linewidth=0.5)
        else:
            signmarkers[ind] = -1
            # Spectral selectivity index - center vs surround means
            plt.subplot(3,3,4)
            plt.scatter(np.nanmean(label_uvpi_surr), np.nanmean(label_uvpi_center), color=[0.1,0.1,0.8], s=2+400*current_n/61, alpha=0.9-uv_sem/4)
            # plt.subplot(338)
            # plt.loglog(tfreq,np.nanmean(label_spectra,axis=(1,2)),color=plot_color,linewidth=0.5)


        # basic AUC scatter
        plt.subplot(3,3,2,aspect=0.6)
        plt.scatter(np.nanmedian(label_BLLauc[0,:]),np.nanmedian(label_LLauc[0,:]), color=[0.2,0.2,0.2], s=20)
        plt.subplot(3,3,3,aspect=0.6)
        plt.scatter(np.nanmedian(label_BLLauc[1,:]),np.nanmedian(label_LLauc[1,:]), color=[0.2,0.2,0.2], s=20)
        # size/class/variance modulated AUC scatter
        # plt.subplot(3,3,2,aspect=0.6)
        # plt.scatter(np.nanmean(label_BLLauc[0,:]),np.nanmean(label_LLauc[0,:]), color=plot_color, s=2+400*current_n/61, alpha=0.9-blue_sem/4)
        # plt.subplot(3,3,3,aspect=0.6)
        # plt.scatter(np.nanmean(label_BLLauc[1,:]),np.nanmean(label_LLauc[1,:]), color=plot_color, s=2+400*current_n/61, alpha=0.9-uv_sem/4)

        # # Integration index - LL+BLL
        # plt.subplot(3,3,5)
        # plt.scatter(np.nanmean(label_prefTFs),np.nanmean(label_LLauc[0,:])+np.nanmean(label_BLLauc[0,:]), color=plot_color, s=2+400*current_n/61, alpha=0.9-blue_sem/5)
        # plt.subplot(3,3,6)
        # plt.scatter(np.nanmean(label_prefTFs),np.nanmean(label_LLauc[1,:])+np.nanmean(label_BLLauc[1,:]), color=plot_color, s=2+400*current_n/61, alpha=0.9-blue_sem/5)

plt.subplot(3,3,5)
plt.ylim(0.01,0.15)
plt.xlabel('Log(Temporal Freq.)')
plt.ylabel('Log(Norm. Power)')
plt.loglog(tfreq,np.nanmean(mean_label_spectrum[:,0,np.logical_and(signmarkers == 1, Nbytype > thresh_n)],axis=(1)),color=[0.8,0.1,0.1],linewidth=2)
plt.loglog(tfreq,np.nanmean(mean_label_spectrum[:,0,np.logical_and(signmarkers == -1, Nbytype > thresh_n)],axis=(1)),color=[0.1,0.1,0.8],linewidth=2)

plt.subplot(3,3,6)
plt.ylim(0.01,0.15)
plt.xlabel('Log(Temporal Freq.)')
plt.loglog(tfreq,np.nanmean(mean_label_spectrum[:,1,np.logical_and(signmarkers == 1, Nbytype > thresh_n)],axis=1),color=[0.8,0.1,0.1],linewidth=2)
plt.loglog(tfreq,np.nanmean(mean_label_spectrum[:,1,np.logical_and(signmarkers == -1, Nbytype > thresh_n)],axis=1),color=[0.1,0.1,0.8],linewidth=2)

plt.subplot(3,3,7)
plt.ylim(0.01,0.15)
plt.ylabel('Log(Norm. Power)')
plt.xlabel('Log(Temporal Freq.)')
plt.loglog(tfreq,np.nanmean(mean_label_spectrum[:,:,np.logical_and(classmarkers == 1, Nbytype > thresh_n)],axis=(1,2)), color=[0,0,0], linewidth=2)
plt.loglog(tfreq,np.nanmean(mean_label_spectrum[:,:,np.logical_and(classmarkers == 1, Nbytype > thresh_n)],axis=(1)), color=[0,0,0], linewidth=0.5, alpha=0.3)

plt.subplot(3,3,8)
plt.ylim(0.01,0.15)
plt.xlabel('Log(Temporal Freq.)')
plt.loglog(tfreq,np.nanmean(mean_label_spectrum[:,:,np.logical_and(classmarkers == 2, Nbytype > thresh_n)],axis=(1,2)), color=[0.2,0.8,0.2], linewidth=2)
plt.loglog(tfreq,np.nanmean(mean_label_spectrum[:,:,np.logical_and(classmarkers == 2, Nbytype > thresh_n)],axis=(1)), color=[0.2,0.8,0.2], linewidth=0.5, alpha=0.5)

plt.subplot(3,3,9)
plt.ylim(0.01,0.15)
plt.xlabel('Log(Temporal Freq.)')
plt.loglog(tfreq,np.nanmean(mean_label_spectrum[:,:,np.logical_and(classmarkers == 3, Nbytype > thresh_n)],axis=(1,2)), color=[0.8,0.2,0.8], linewidth=2)
plt.loglog(tfreq,np.nanmean(mean_label_spectrum[:,:,np.logical_and(classmarkers == 3, Nbytype > thresh_n)],axis=(1)), color=[0.8,0.2,0.8], linewidth=0.5, alpha=0.4)

# save it
fname = '/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/label summaries/CENTERS_auc_spectrum_byType.pdf'
tempfig.savefig(fname, format='pdf', orientation='portrait')
#%% spectral preference index by cell type
uvpifig = plt.figure(figsize=(15, 6))
plt.suptitle('Spectral Selectivity by Cell Type (N>=3)',fontsize='xx-large', fontweight='bold')

# remove 0-value means and associated cell types
thresh_types = unique_types[np.nonzero(uvpi_means)]
thresh_Nbytype = Nbytype[np.nonzero(uvpi_means)]
thresh_uvpi = uvpi_means[np.nonzero(uvpi_means)]

# sort by means
sorted_inds = np.argsort(thresh_uvpi)
sorted_types = thresh_types[sorted_inds]
sorted_Nbytype = thresh_Nbytype[sorted_inds]

for n in range(0,len(sorted_types)):
    current_label = sorted_types[n]
    label_uvpi_center = uvpi_center[scraped_log[:,14] == current_label]
    # remove NaNs
    label_uvpi_center = label_uvpi_center[~np.isnan(label_uvpi_center)]
    # plt.boxplot(label_uvpi_center, positions=[n], widths=0.8, showfliers=False, showcaps=False, whis=[0,100], labels={''})
    plt.scatter(n*np.ones(len(label_uvpi_center)), label_uvpi_center, color=[0,0,0], s=8, alpha=0.3)
    # plt.violinplot(label_uvpi_center, positions=[n], showextrema=False, points=200)
    plt.text(n-0.2, -1.35, sorted_types[n], rotation='vertical')
    plt.text(n-0.3, 1.15, str(len(label_uvpi_center)))

# plot means and plot params
plt.scatter(np.arange(0,len(thresh_types),1), thresh_uvpi[sorted_inds], color=[0,0,0], s=30)
plt.plot([-1,len(thresh_types)+1], [0,0], 'k:')
plt.ylabel('Spectral Preference Index')

# save it
fname = '/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/label summaries/Spectral_Preference_byType.pdf'
uvpifig.savefig(fname, format='pdf', orientation='landscape')


#%% By cell type / advanced temporal summary for RF surrounds

# this value determines the threshold for N that will be required to include a cell type in summary plots (non-inclusive lower bound)
thresh_n = 2

utype_indices = np.unique(scraped_log[:,14]) != ''
unique_types = np.unique(scraped_log[:,14])[utype_indices]
classmarkers = np.zeros(len(unique_types))
signmarkers = np.zeros(len(unique_types))
Nbytype = np.zeros(len(unique_types))

mean_label_spectrum = np.zeros((psdt.shape[0],2,len(unique_types)))

tempfig = plt.figure(figsize=(15, 15))
plt.suptitle('Temporal Summary for RF Surrouds, by Cell Type (N>=3)',fontsize='xx-large', fontweight='bold')
# build axes
plt.subplot(3,3,1,aspect=0.6)
plt.plot([-20,20],[0,0],'k--',linewidth=0.5)
plt.plot([0,0],[-30,30],'k--',linewidth=0.5)
plt.plot([-20,20],[20,-20],'k--',linewidth=0.5)
plt.xlim(-20,20)
plt.ylim(-30,30)
plt.ylabel('Last Lobe AUC')
plt.xlabel('Before Last Lobe AUC')
plt.text(-15,25,'ON', fontsize='large', fontweight='demibold')
plt.text(-8,-9,'Differentiators', fontsize='large', fontweight='demibold', rotation=-30)
plt.text(10,-27,'OFF', fontsize='large', fontweight='demibold')
plt.text(-7,-24,'Integrators', fontsize='large', fontweight='demibold')
plt.text(-7,22,'Integrators', fontsize='large', fontweight='demibold')
plt.text(-17,1,'Delayed', fontsize='large', fontweight='demibold')
plt.text(-19,-3,'Integrators', fontsize='large', fontweight='demibold')
plt.text(8,1,'Delayed', fontsize='large', fontweight='demibold')
plt.text(6,-3,'Integrators', fontsize='large', fontweight='demibold')

plt.subplot(3,3,2,aspect=0.8)
plt.plot([-0.6,0.6],[0,0],'k--',linewidth=0.5)
plt.plot([0,0],[-0.6,0.6],'k--',linewidth=0.5)
plt.plot([-0.6,0.6],[0.6,-0.6],'k--',linewidth=0.5)
plt.xlim(-0.6,0.6)
plt.ylim(-0.6,0.6)
plt.title('Blue')
plt.ylabel('Last Lobe AUC')
plt.xlabel('Before Last Lobe AUC')
plt.subplot(3,3,3,aspect=0.8)
plt.plot([-0.6,0.6],[0,0],'k--',linewidth=0.5)
plt.plot([0,0],[-0.6,0.6],'k--',linewidth=0.5)
plt.plot([-0.6,0.6],[0.6,-0.6],'k--',linewidth=0.5)
plt.xlim(-0.6,0.6)
plt.ylim(-0.6,0.6)
plt.title('UV')
plt.ylabel('Last Lobe AUC')
plt.xlabel('Before Last Lobe AUC')

plt.subplot(3,3,4)
plt.xlim(-0.5,3)
plt.ylim(-1.2,1.2)
plt.plot([-20,20],[0,0],'k--',linewidth=0.5)
plt.ylabel('UV Preference Index')
plt.xlabel('Preferred TF')




for ind in range(0,len(unique_types)):
    # pull summary measurements for current cell type
    current_label = unique_types[ind]
    current_cells = scraped_log[scraped_log[:,14] == current_label,1]
    current_n = len(current_cells)
    Nbytype[ind] = current_n

    if current_n > thresh_n:
        # temporal spectrum
        label_spectra = psdt_surr[:,:,scraped_log[:,14] == current_label]
        mean_label_spectrum[:,:,ind] = np.nanmean(label_spectra,axis=2)
        # preferred freq
        label_prefTFs = pref_tfs[scraped_log[:,14] == current_label]
        # TFmeans[ind,2] = np.nanmean(label_prefTFs)
        # AUCs
        label_BLLauc = surr_before_zc_auc[:,scraped_log[:,14] == current_label]
        label_LLauc = surr_after_zc_auc[:,scraped_log[:,14] == current_label]
        blue_sem = np.mean([np.std(label_BLLauc[0,:])/np.sqrt(current_n),np.std(label_LLauc[0,:])/np.sqrt(current_n)])
        uv_sem = np.mean([np.std(label_BLLauc[1,:])/np.sqrt(current_n),np.std(label_LLauc[1,:])/np.sqrt(current_n)])

        # define plot color based on cell class, and tag marker vector
        if np.logical_or(np.logical_or(current_label[0]=='P',current_label[0]=='D'),current_label[0]=='S'):
            plot_color=[0.8,0.2,0.8]
            classmarkers[ind] = 3
            # plt.subplot(339)
            # plt.loglog(tfreq,np.nanmean(label_spectra,axis=(1,2)),color=plot_color,linewidth=0.5)
        elif np.logical_or(current_label[0]=='C',current_label[0]=='Y'):
            plot_color=[0.2,0.8,0.2]
            classmarkers[ind] = 2
            # plt.subplot(338)
            # plt.loglog(tfreq,np.nanmean(label_spectra,axis=(1,2)),color=plot_color,linewidth=0.5)
        else:
            plot_color=[0,0,0]
            classmarkers[ind] = 1
            # plt.subplot(337)
            # plt.loglog(tfreq,np.nanmean(label_spectra,axis=(1,2)),color=plot_color,linewidth=0.5)
        # manual correction for LMa's and Olt
        if np.logical_or(current_label[0:1]=='LM',current_label[0:1]=='Ol'):
            plot_color=[0.8,0.2,0.8]
            classmarkers[ind] = 3
            # plt.subplot(339)
            # plt.loglog(tfreq,np.nanmean(label_spectra,axis=(1,2)),color=plot_color,linewidth=0.5)

        # plot spectra based on sign of last lobe AUC
        if np.sign(np.nanmedian(label_LLauc)) == 1:
            signmarkers[ind] = 1
            # Spectral selectivity index - AUC diff blue-UV
            # plt.subplot(3,3,4)
            # plt.scatter(np.nanmean(label_prefTFs),(np.nanmean(label_LLauc[1,:])-np.nanmean(label_LLauc[0,:]))/(np.nanmean(label_LLauc[1,:])+np.nanmean(label_LLauc[0,:])), color=[0.8,0.1,0.1], s=2+400*current_n/61, alpha=0.9-uv_sem/3)
            # plt.subplot(337)
            # plt.loglog(tfreq,np.nanmean(label_spectra,axis=(1,2)),color=plot_color,linewidth=0.5)
        else:
            signmarkers[ind] = -1
            # Spectral selectivity index - AUC diff blue-UV
            # plt.subplot(3,3,4)
            # plt.scatter(np.nanmean(label_prefTFs),(np.nanmean(label_LLauc[1,:])-np.nanmean(label_LLauc[0,:]))/(np.nanmean(label_LLauc[1,:])+np.nanmean(label_LLauc[0,:])), color=[0.1,0.1,0.8], s=2+400*current_n/61, alpha=0.9-uv_sem/3)
            # plt.subplot(338)
            # plt.loglog(tfreq,np.nanmean(label_spectra,axis=(1,2)),color=plot_color,linewidth=0.5)

        # basic AUC scatter
        plt.subplot(3,3,2,aspect=0.8)
        plt.scatter(np.nanmedian(label_BLLauc[0,:]),np.nanmedian(label_LLauc[0,:]), color=[0.2,0.2,0.2], s=20)
        plt.subplot(3,3,3,aspect=0.8)
        plt.scatter(np.nanmedian(label_BLLauc[1,:]),np.nanmedian(label_LLauc[1,:]), color=[0.2,0.2,0.2], s=20)
        # size/class/variance modulated AUC scatter
        # plt.subplot(3,3,2,aspect=0.8)
        # plt.scatter(np.nanmean(label_BLLauc[0,:]),np.nanmean(label_LLauc[0,:]), color=plot_color, s=2+400*current_n/61, alpha=0.9-blue_sem/4)
        # plt.subplot(3,3,3,aspect=0.8)
        # plt.scatter(np.nanmean(label_BLLauc[1,:]),np.nanmean(label_LLauc[1,:]), color=plot_color, s=2+400*current_n/61, alpha=0.9-uv_sem/4)


        # # Integration index - LL+BLL
        # plt.subplot(3,3,5)
        # plt.scatter(np.nanmean(label_prefTFs),np.nanmean(label_LLauc[0,:])+np.nanmean(label_BLLauc[0,:]), color=plot_color, s=2+400*current_n/61, alpha=0.9-2.5*blue_sem)
        # plt.subplot(3,3,6)
        # plt.scatter(np.nanmean(label_prefTFs),np.nanmean(label_LLauc[1,:])+np.nanmean(label_BLLauc[1,:]), color=plot_color, s=2+400*current_n/61, alpha=0.9-2.5*blue_sem)

plt.subplot(3,3,5)
plt.ylim(0.01,0.15)
plt.xlabel('Log(Temporal Freq.)')
plt.ylabel('Log(Norm. Power)')
plt.loglog(tfreq,np.nanmean(mean_label_spectrum[:,0,np.logical_and(signmarkers == 1, Nbytype > thresh_n)],axis=(1)),color=[0.8,0.1,0.1],linewidth=2)
plt.loglog(tfreq,np.nanmean(mean_label_spectrum[:,0,np.logical_and(signmarkers == -1, Nbytype > thresh_n)],axis=(1)),color=[0.1,0.1,0.8],linewidth=2)

plt.subplot(3,3,6)
plt.ylim(0.01,0.15)
plt.xlabel('Log(Temporal Freq.)')
plt.loglog(tfreq,np.nanmean(mean_label_spectrum[:,1,np.logical_and(signmarkers == 1, Nbytype > thresh_n)],axis=1),color=[0.8,0.1,0.1],linewidth=2)
plt.loglog(tfreq,np.nanmean(mean_label_spectrum[:,1,np.logical_and(signmarkers == -1, Nbytype > thresh_n)],axis=1),color=[0.1,0.1,0.8],linewidth=2)

plt.subplot(3,3,7)
plt.ylim(0.01,0.15)
plt.ylabel('Log(Norm. Power)')
plt.xlabel('Log(Temporal Freq.)')
plt.loglog(tfreq,np.nanmean(mean_label_spectrum[:,:,np.logical_and(classmarkers == 1, Nbytype > thresh_n)],axis=(1,2)), color=[0,0,0], linewidth=2)
plt.loglog(tfreq,np.nanmean(mean_label_spectrum[:,:,np.logical_and(classmarkers == 1, Nbytype > thresh_n)],axis=(1)), color=[0,0,0], linewidth=0.5, alpha=0.3)

plt.subplot(3,3,8)
plt.ylim(0.01,0.15)
plt.xlabel('Log(Temporal Freq.)')
plt.loglog(tfreq,np.nanmean(mean_label_spectrum[:,:,np.logical_and(classmarkers == 2, Nbytype > thresh_n)],axis=(1,2)), color=[0.2,0.8,0.2], linewidth=2)
plt.loglog(tfreq,np.nanmean(mean_label_spectrum[:,:,np.logical_and(classmarkers == 2, Nbytype > thresh_n)],axis=(1)), color=[0.2,0.8,0.2], linewidth=0.5, alpha=0.5)

plt.subplot(3,3,9)
plt.ylim(0.01,0.15)
plt.xlabel('Log(Temporal Freq.)')
plt.loglog(tfreq,np.nanmean(mean_label_spectrum[:,:,np.logical_and(classmarkers == 3, Nbytype > thresh_n)],axis=(1,2)), color=[0.8,0.2,0.8], linewidth=2)
plt.loglog(tfreq,np.nanmean(mean_label_spectrum[:,:,np.logical_and(classmarkers == 3, Nbytype > thresh_n)],axis=(1)), color=[0.8,0.2,0.8], linewidth=0.5, alpha=0.4)

# save it
fname = '/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/label summaries/SURROUNDS_auc_spectrum_byType.pdf'
tempfig.savefig(fname, format='pdf', orientation='portrait')
#%% Center vs. Surround temporal comparison

thresh_n = 1

utype_indices = np.unique(scraped_log[:,14]) != ''
unique_types = np.unique(scraped_log[:,14])[utype_indices]


mean_label_spectrum = np.zeros((psdt.shape[0],2,len(unique_types)))
mean_surround_spectrum = np.zeros((psdt_surr.shape[0],2,len(unique_types)))
mean_LL = np.zeros((2,len(unique_types)))
mean_surr_LL = np.zeros((2,len(unique_types)))

tempfig = plt.figure(figsize=(15, 15))
plt.suptitle('Center vs. Surround by Cell Type (N>=3)',fontsize='xx-large', fontweight='bold')
for ind in range(0,len(unique_types)):
    # pull summary measurements for current cell type
    current_label = unique_types[ind]
    current_cells = scraped_log[scraped_log[:,14] == current_label,1]
    current_n = len(current_cells)
    Nbytype[ind] = current_n

    if current_n > thresh_n:
        # temporal spectra
        label_spectra = psdt[:,:,scraped_log[:,14] == current_label]
        label_surr_spectra = psdt_surr[:,:,scraped_log[:,14] == current_label]
        mean_label_spectrum[:,:,ind] = np.nanmean(label_spectra,axis=2)
        mean_surround_spectrum[:,:,ind] = np.nanmean(label_surr_spectra,axis=2)
        # AUCs
        label_BLLauc = before_zc_auc[:,scraped_log[:,14] == current_label]
        label_LLauc = after_zc_auc[:,scraped_log[:,14] == current_label]
        mean_LL[:,ind] = np.nanmean(label_LLauc,axis=1)
        label_surr_BLLauc = surr_before_zc_auc[:,scraped_log[:,14] == current_label]
        label_surr_LLauc = surr_after_zc_auc[:,scraped_log[:,14] == current_label]
        mean_surr_LL[:,ind] = np.nanmean(label_surr_LLauc,axis=1)

        # define plot color based on cell class, and tag marker vector
        if np.logical_or(np.logical_or(current_label[0]=='P',current_label[0]=='D'),current_label[0]=='S'):
            plot_color=[0.8,0.2,0.8]
            classmarkers[ind] = 3
        elif np.logical_or(current_label[0]=='C',current_label[0]=='Y'):
            plot_color=[0.2,0.8,0.2]
            classmarkers[ind] = 2
        else:
            plot_color=[0,0,0]
            classmarkers[ind] = 1
        # manual correction for LMa's and Olt
        if np.logical_or(current_label[0:1]=='LM',current_label[0:1]=='Ol'):
            plot_color=[0.8,0.2,0.8]
            classmarkers[ind] = 3

        # plot last lobe AUCs for center vs. surround
        plt.subplot(3,3,1)
        plt.scatter(np.nanmean(label_surr_LLauc[0,:]),np.nanmean(label_LLauc[0,:]), color=plot_color)
        plt.subplot(3,3,2)
        plt.scatter(np.nanmean(label_surr_LLauc[1,:]),np.nanmean(label_LLauc[1,:]), color=plot_color)
        plt.subplot(3,3,7)
        plt.scatter(np.nanmean(label_surr_BLLauc[0,:]),np.nanmean(label_BLLauc[0,:]), color=plot_color)
        plt.subplot(3,3,8)
        plt.scatter(np.nanmean(label_surr_BLLauc[1,:]),np.nanmean(label_BLLauc[1,:]), color=plot_color)

# correlation tables
auc_corr_table = np.corrcoef(np.append(mean_LL,mean_surr_LL,axis=0))
bxcorr = np.corrcoef(np.append(mean_label_spectrum[:,0,Nbytype>thresh_n], mean_surround_spectrum[:,0,Nbytype>thresh_n],axis=0))[0:30,30:]
uvxcorr = np.corrcoef(np.append(mean_label_spectrum[:,1,Nbytype>thresh_n], mean_surround_spectrum[:,1,Nbytype>thresh_n],axis=0))[0:30,30:]

# build axes
plt.subplot(3,3,1)
plt.plot([-20,20],[0,0],'k--',linewidth=0.5)
plt.plot([0,0],[-30,30],'k--',linewidth=0.5)
plt.xlim(-1,1)
plt.ylim(-12,12)
plt.text(0.5,9,'R = '+str(np.round(auc_corr_table[0,2],2)))
plt.ylabel('Center Last Lobe AUC')
plt.xlabel('Surround Last Lobe AUC')
plt.title('Blue')
plt.subplot(3,3,2)
plt.plot([-20,20],[0,0],'k--',linewidth=0.5)
plt.plot([0,0],[-30,30],'k--',linewidth=0.5)
plt.xlim(-1,1)
plt.ylim(-12,12)
plt.text(0.5,9,'R = '+str(np.round(auc_corr_table[1,3],2)))
plt.ylabel('Center Last Lobe AUC')
plt.xlabel('Surround Last Lobe AUC')
plt.title('UV')

plt.subplot(3,3,4)
x = plt.imshow(bxcorr, clim=[-0.8,0.8])
x.axes.get_yaxis().set_ticks(np.arange(0,30,1))
x.axes.get_yaxis().set_ticklabels(np.round(tfreq,2).astype('str'))
x.axes.get_xaxis().set_ticks(np.arange(0,30,1))
x.axes.get_xaxis().set_ticklabels(np.round(tfreq,2).astype('str'), rotation='vertical')
plt.ylabel('Surround Norm. Power')
plt.xlabel('Center Norm. Power')

plt.subplot(3,3,5)
x = plt.imshow(uvxcorr, clim=[-0.8,0.8])
x.axes.get_yaxis().set_ticks([])
x.axes.get_xaxis().set_ticks(np.arange(0,30,1))
x.axes.get_xaxis().set_ticklabels(np.round(tfreq,2).astype('str'), rotation='vertical')
plt.xlabel('Center Norm. Power')

plt.subplot(3,3,6)
plt.axis('off')
plt.colorbar(location='left', shrink=0.8, aspect=10)

plt.subplot(3,3,7)
plt.plot([-20,20],[0,0],'k--',linewidth=0.5)
plt.plot([0,0],[-30,30],'k--',linewidth=0.5)
plt.xlim(-1,1)
plt.ylim(-8,8)
plt.ylabel('Center Before Last Lobe AUC')
plt.xlabel('Surround Before Last Lobe AUC')
plt.subplot(3,3,8)
plt.plot([-20,20],[0,0],'k--',linewidth=0.5)
plt.plot([0,0],[-30,30],'k--',linewidth=0.5)
plt.xlim(-1,1)
plt.ylim(-8,8)
plt.ylabel('Center Before Last Lobe AUC')
plt.xlabel('Surround Before Last Lobe AUC')

# save it
fname = '/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/label summaries/center_vs_surround_temporal_dynamics.pdf'
tempfig.savefig(fname, format='pdf', orientation='portrait')


# %% Flicker summary plots - all cells

culled_flick_mds = flick_mod_depths[:,~np.isnan(flick_mod_depths[0,:])]
culled_flick_DCs = flick_DCs[:,~np.isnan(flick_DCs[0,:])]
# sort by response vector length
# sorted_flick_mds = culled_flick_mds[:,np.argsort(np.dot(np.transpose(culled_flick_mds),[1,2,3,4]))]
sorted_flick_mds = culled_flick_mds[:,np.argsort(culled_flick_mds[0,:])]
sorted_flick_DCs = culled_flick_DCs[:,np.argsort(culled_flick_mds[0,:])]

# sort by response vector angle
# flick_md_sort_weights = np.average(np.transpose(np.tile([1,2,3,4],(culled_flick_mds.shape[1],1))), axis=0, weights=culled_flick_mds)
# sorted_flick_mds = culled_flick_mds[:,np.argsort(flick_md_sort_weights)]

flkfig=plt.figure(figsize=(15, 8))
plt.subplot(1,2,1)
x = plt.imshow(np.transpose(sorted_flick_mds), aspect=0.05, clim=[0,2])
x.axes.get_xaxis().set_ticks([0,1,2,3])
x.axes.get_xaxis().set_ticklabels(['0.1','0.5','1','2'])
plt.ylabel('Cell')
plt.xlabel('Flicker Frequency (Hz)')
plt.title('Flicker Modulation Depth')
plt.colorbar(fraction=0.1, aspect=4, ticks=[0,1,2,3], label='x100 ''%'' dF/F', shrink=0.5);
plt.subplot(1,2,2)
x = plt.imshow(np.transpose(sorted_flick_DCs), aspect=0.05, cmap='PiYG', clim=[-1,1])
x.axes.get_xaxis().set_ticks([0,1,2,3])
x.axes.get_xaxis().set_ticklabels(['0.1','0.5','1','2'])
plt.ylabel('Cell')
plt.xlabel('Flicker Frequency (Hz)')
plt.title('Flicker DC offset')
plt.colorbar(fraction=0.1, aspect=4, ticks=[-1,0,1], label='x100 ''%'' dF/F', shrink=0.5);

# save it
fname = '/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all cell summaries/flicker_summary.pdf'
flkfig.savefig(fname, format='pdf', orientation='landscape')
#%% flicker vs. temporal figure

thresh_n = 2

flkfig=plt.figure(figsize=(15, 12))

ps_val_means = np.zeros((4,len(unique_types)))
md_means = np.zeros((4,len(unique_types)))
dc_means = np.zeros((4,len(unique_types)))

for ind in range(0,len(unique_types)):
    # pull summary measurements for current cell type
    current_label = unique_types[ind]
    current_cells = scraped_log[scraped_log[:,14] == current_label,1]
    current_n = len(current_cells)

    if current_n > thresh_n:
        # temporal spectrum
        label_spectra = psdt[:,:,scraped_log[:,14] == current_label]
        # flicker modulation depth and DC signal
        label_mds = flick_mod_depths[:,scraped_log[:,14] == current_label]
        label_dcs = flick_DCs[:,scraped_log[:,14] == current_label]

        # grab spectrum points corresponding to flick stimuli
        ps_vals = np.zeros(4)
        ps_vals[0] = np.nanmean(label_spectra[0:2,:,:],axis=(0,1,2))
        ps_vals[1] = np.nanmean(label_spectra[1:3,:,:],axis=(0,1,2))
        ps_vals[2] = np.nanmean(label_spectra[3,:,:],axis=(0,1))
        ps_vals[3] = np.nanmean(label_spectra[6,:,:],axis=(0,1))

        # save means to vecs for corr calcs
        ps_val_means[:,ind] = ps_vals
        md_means[:,ind] = np.nanmean(label_mds,axis=1)
        dc_means[:,ind] = np.nanmean(label_dcs,axis=1)

        # calculate SEMs for alpha
        md_sem = np.nanstd(label_mds,axis=1)/np.sqrt(current_n)
        dc_sem = np.nanstd(label_dcs,axis=1)/np.sqrt(current_n)

        # define plot color by cell class
        # if classmarkers[ind] == 1:
        #     plot_color=[0,0,0]
        # elif classmarkers[ind] == 2:
        #     plot_color=[0.2,0.8,0.2]
        # elif classmarkers[ind] == 3:
        #     plot_color=[0.8,0.2,0.8]

        # define plot color by cell sign
        if signmarkers[ind] == 1:
            plot_color=[0.8,0.1,0.1]
        elif signmarkers[ind] == -1:
            plot_color=[0.1,0.1,0.8]

        # plot
        for n in range(0,4):
            plt.subplot(3,4,(n+1)), plt.scatter(np.nanmean(label_mds,axis=1)[n], ps_vals[n], color=plot_color, s=2+400*current_n/61, alpha=0.8-md_sem[n])
            plt.subplot(3,4,4+(n+1)), plt.scatter(np.nanmean(label_dcs,axis=1)[n], ps_vals[n], color=plot_color, s=2+400*current_n/61, alpha=0.8-dc_sem[n])
            plt.subplot(3,4,8+(n+1)), plt.scatter(np.nanmean(label_dcs,axis=1)[n], np.nanmean(label_mds,axis=1)[n], color=plot_color, s=2+400*current_n/61, alpha=0.8-md_sem[n])
corrmat = np.corrcoef(np.append(ps_val_means[:,Nbytype>thresh_n], np.append(md_means[:,Nbytype>thresh_n],dc_means[:,Nbytype>thresh_n],axis=0),axis=0))

plt.subplot(341)
plt.ylim(0,0.12)
plt.xlim(0,2)
plt.text(1.2,0.01,'R = ' + str(np.round(corrmat[0,4],3)))
plt.title('0.1 Hz')
plt.xlabel('Flicker Modulation Depth')
plt.ylabel('Freq-matched Power')
plt.subplot(342)
plt.ylim(0,0.12)
plt.xlim(0,2)
plt.text(1.2,0.01,'R = ' + str(np.round(corrmat[1,5],3)))
plt.title('0.5 Hz')
plt.subplot(343)
plt.ylim(0,0.12)
plt.xlim(0,2)
plt.text(1.2,0.01,'R = ' + str(np.round(corrmat[2,6],3)))
plt.title('1 Hz')
plt.subplot(344)
plt.ylim(0,0.12)
plt.xlim(0,2)
plt.text(1.2,0.01,'R = ' + str(np.round(corrmat[3,7],3)))
plt.title('2 Hz')

plt.subplot(345)
plt.ylim(0,0.12)
plt.xlim(-0.6,0.6)
plt.xlabel('Flicker DC Offset')
plt.ylabel('Freq-matched Power')
plt.subplot(346)
plt.ylim(0,0.12)
plt.xlim(-0.6,0.6)
plt.subplot(347)
plt.ylim(0,0.12)
plt.xlim(-0.6,0.6)
plt.subplot(348)
plt.ylim(0,0.12)
plt.xlim(-0.6,0.6)

plt.subplot(349)
plt.ylim(0,2)
plt.xlim(-0.6,0.6)
plt.text(0.1,1.75,'R = ' + str(np.round(corrmat[4,8],3)))
plt.xlabel('Flicker DC Offset')
plt.ylabel('Flicker Modulation Depth')
plt.subplot(3,4,10)
plt.ylim(0,2)
plt.xlim(-0.6,0.6)
plt.text(0.1,1.75,'R = ' + str(np.round(corrmat[5,9],3)))
plt.subplot(3,4,11)
plt.ylim(0,2)
plt.xlim(-0.6,0.6)
plt.text(0.1,1.75,'R = ' + str(np.round(corrmat[6,10],3)))
plt.subplot(3,4,12)
plt.ylim(0,2)
plt.xlim(-0.6,0.6)
plt.text(0.1,1.75,'R = ' + str(np.round(corrmat[7,11],3)))

# save it
fname = '/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/label summaries/flicker_correlations_byType.pdf'
flkfig.savefig(fname, format='pdf', orientation='landscape')





#%% spatial summary figure
# pull list of unique cell type labels
utype_indices = np.unique(scraped_log[:,14]) != ''
unique_types = np.unique(scraped_log[:,14])[utype_indices]

spacefig = plt.figure(figsize=(15, 12))
plt.suptitle('Spatial Summary - All Neurons',fontsize='xx-large', fontweight='bold')

# can consider fitting exponential or lognormal distr. to these
plt.subplot(3,3,1)
os_hist = np.histogram(orient_selects,bins=16,range=(0.5,3.5))
os_norm = os_hist[0]/np.sum(os_hist[0])
plt.ylabel('Probability')
plt.xlabel('Orientation Selectivity Ratio')
plt.plot(os_hist[1][:16],os_norm);

plt.subplot(3,3,2)
ca_hist = np.histogram(center_areas,bins=19,range=(0,900))
ca_norm = ca_hist[0]/np.sum(ca_hist[0])
plt.plot(ca_hist[1][1:],ca_norm);
plt.ylabel('Probability')
plt.xlabel('RF Center Area (sq. deg)')
el_hist = np.histogram(ellipse_areas,bins=19,range=(0,900))
el_norm = el_hist[0]/np.sum(el_hist[0])
plt.plot(el_hist[1][1:],el_norm);

plt.subplot(3,3,3)
ds_hist = np.histogram(ds_index,bins=12,range=(0,1))
ds_norm = ds_hist[0]/np.sum(ds_hist[0])
plt.plot(ds_hist[1][1:],ds_norm);
plt.ylabel('Probability')
plt.xlabel('DS Index')

plt.subplot(3,3,4)
sf_vec = freq[1:]
norm_ns_ps = (1/freq[1:])/np.sum(1/freq[1:])
mean_spatial_ps = np.nanmean(psd1D,axis=(1,2))

mean_label_spectrum = np.zeros((psd1D.shape[0],len(unique_types)))
# for ind in range(0,1):
for ind in range(0,len(unique_types)):
    current_label = unique_types[ind]
    label_spectra = psd1D[:,:,scraped_log[:,14] == current_label]
    mean_label_spectrum[:,ind] = np.nanmean(label_spectra,axis=(1,2))
    plt.loglog(sf_vec[:15],mean_label_spectrum[:15,ind],color=[0.4,0.4,0.4],linewidth=0.5)
plt.loglog(sf_vec[:15],mean_spatial_ps[:15],linewidth=2)
plt.loglog(sf_vec[:15],norm_ns_ps[:15],'k-')
plt.ylabel('Log(Norm. Power)')
plt.xlabel('Log(Spatial Freq) (cycles per deg)')

plt.subplot(3,3,5)
sf_hist = np.histogram(pref_sfs,bins=12,range=(0,20/180))
sf_norm = sf_hist[0]/np.sum(sf_hist[0])
plt.plot(sf_hist[1][1:],sf_norm);
plt.ylabel('Probability')
plt.xlabel('Preferred Spatial Freq (cycles per deg)')

# initialize sorting array for mean displacements and collective culled vector
label_mean_os = np.zeros(len(unique_types))

# replace nans in orientation selectivity vector with 1s (i.e., no selectivity)
# orient_selects = np.where(np.isnan(orient_selects),1,orient_selects)

# initializae variance by cell type array
po_var = np.zeros(len(unique_types))

area_thresh = 105
thresh_n = 2

# sort by mean os
for n in range(0,len(unique_types)):
    label = unique_types[n]
    label_os = orient_selects[scraped_log[:,14] == label]
    label_areas = ellipse_areas[scraped_log[:,14] == label]
    label_uvpi = uvpi_center[scraped_log[:,14] == label]
    # NaN-out OS values for cells with area less than area_thresh
    label_os[label_areas < area_thresh] = np.nan
    # take mean
    label_mean_os[n] = np.nanmedian(label_os)
    # preferred orientation variance
    label_pref_orients = pref_orients[scraped_log[:,14] == label]
    po_var[n] = stats.circvar(np.deg2rad(label_pref_orients), high=2*np.pi, low=0, nan_policy='omit')
sorted_types = unique_types[np.argsort(label_mean_os)]
sorted_po_var = po_var[np.argsort(label_mean_os)]

# plot
counter = 0
for n in range(0,len(sorted_types)):
    label = sorted_types[n]
    if np.logical_or(np.logical_or(np.logical_or(label=='Tm5b',label=='Tm29'),label=='Dm9'),label=='TmY18'):
        pass
    else:
        label_os = orient_selects[scraped_log[:,14] == label]
        label_areas = ellipse_areas[scraped_log[:,14] == label]
        label_uvpi = uvpi_center[scraped_log[:,14] == label]
        # NaN-out OS values for cells with area less than area_thresh
        label_os[label_areas < area_thresh] = np.nan
        # remove NaNs
        label_os = label_os[~np.isnan(label_os)]
        if len(label_os) > thresh_n:
            counter = counter + 1
            if label[0:3] == 'Dm3':
                plotcolor = [0.2,0.2,0.8]
            elif label == 'Dm16':
                plotcolor = [0.4,0.8,0.4]
            elif label == 'TmY9a':
                plotcolor = [0.8,0.4,0.4]
            elif label == 'T4':
                plotcolor = [0.8,0.8,0.4]
            elif label == 'TmY16':
                plotcolor = [0.8,0.4,0.8]
            # elif label == 'L2':
            #     plotcolor = [0.8,0.6,0.2]
            else:
                plotcolor = [0.4,0.4,0.4]
            plt.subplot(3,1,3)
            # if len(label_os) >= 6:
            #     plt.boxplot(label_os, positions=[counter], widths=0.8, showfliers=False, showcaps=False, whis=[0,100], labels={''})
            # else:
            #     plt.plot([counter, counter], [np.max(label_os), np.min(label_os)], color='k', linewidth=0.75)
            plt.scatter(np.full(len(label_os),counter),label_os,color=[0,0,0],s=8,alpha=0.3)
            plt.scatter(counter,np.nanmedian(label_os),color=[0,0,0],s=30)
            plt.text(counter-0.2, -0.4, sorted_types[n], rotation='vertical')
            plt.text(counter-0.3, 0.6, str(len(label_os)))
            plt.subplot(3,3,6)
            plt.scatter(np.nanmedian(label_os),sorted_po_var[n],color=plotcolor,s=20)

# plt.scatter(center_areas,orient_selects,s=2)
plt.subplot(3,1,3)
plt.ylim(0.4,3.5)
# plt.xlim(0.5,4.5)
plt.ylabel('Orientation Selectivity Ratio')
plt.subplot(3,3,6)
plt.xlim(1.1,2.3)
plt.xlabel('Orientation Selectivity Ratio')
plt.ylabel('Variance of Preferred Orientations')

# remove nans by overwriting with the median nonzero variance (this preserves 0 as a "special" result for this metric), then save circular variance by cell type
# po_var[np.isnan(po_var)] = np.nanmedian(po_var[np.nonzero(po_var)])
# np.save('/Volumes/TBD/Bruker/STRFs/compiled/pref_orientation_variance_byType.npy', po_var)

# save figure
fname = '/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all cell summaries/spatial_summary.pdf'
spacefig.savefig(fname, format='pdf', orientation='portrait')



#%% by cell type / advanced spatial summary
thresh_n = 2

utype_indices = np.unique(scraped_log[:,14]) != ''
unique_types = np.unique(scraped_log[:,14])[utype_indices]

mean_label_spectra = np.zeros((psd1D.shape[0],2,len(unique_types)))
label_os_mean = np.zeros(len(unique_types))
label_ser_mean = np.zeros(len(unique_types))
label_os_sem = np.zeros(len(unique_types))
label_ser_sem = np.zeros(len(unique_types))
areas_bytype = np.zeros(len(unique_types))

spacefig=plt.figure(figsize=(15, 12))
plt.suptitle('Spatial Summary by Type (N>=3)',fontsize='xx-large', fontweight='bold')

for ind in range(0,len(unique_types)):
    # pull summary measurements for current cell type
    current_label = unique_types[ind]
    current_cells = scraped_log[scraped_log[:,14] == current_label,1]
    current_n = len(current_cells)

    if current_n > thresh_n:
        # spatial spectrum
        label_spectra = psd1D[:,:,scraped_log[:,14] == current_label]
        mean_label_spectra[:,:,ind] = np.nanmean(label_spectra,axis=2) # second dim is color
        # preferred spatial frequency
        label_prefSFs = pref_sfs[scraped_log[:,14] == current_label]
        # OS - preference index
        label_os = orient_selects[scraped_log[:,14] == current_label]
        label_os_mean[ind] = np.nanmean(label_os)
        label_os_sem[ind] = np.nanstd(label_os)/np.sqrt(Nbytype[ind])
        # OS - spatial extent ratio long::short
        label_ser = long_axes[scraped_log[:,14] == current_label]/short_axes[scraped_log[:,14] == current_label]
        label_ser_mean[ind] = np.nanmean(label_ser)
        label_ser_sem[ind] = np.nanstd(label_ser)/np.sqrt(Nbytype[ind])
        # RF area
        label_area = center_areas[scraped_log[:,14] == current_label]
        label_area_mean = np.nanmean(label_area)
        ellipse_area = ellipse_areas[scraped_log[:,14] == current_label]
        ellipse_area_mean = np.nanmean(ellipse_area)
        areas_bytype[ind] = ellipse_area_mean

        # define plot color by cell class
        if classmarkers[ind] == 1:
            plot_color=[0,0,0]
            plt.subplot(3,3,1)
            plt.scatter(np.nanmean(label_prefSFs), label_os_mean[ind], color=plot_color, s=2+400*current_n/61, alpha=0.8-label_os_sem[ind])
            plt.subplot(3,3,4)
            plt.scatter(ellipse_area_mean, label_os_mean[ind], color=plot_color, s=2+400*current_n/61, alpha=0.8-label_os_sem[ind])
        elif classmarkers[ind] == 2:
            plot_color=[0.2,0.8,0.2]
            plt.subplot(3,3,1)
            plt.scatter(np.nanmean(label_prefSFs), label_os_mean[ind], color=plot_color, s=2+400*current_n/61, alpha=0.8-label_os_sem[ind])
            plt.subplot(3,3,4)
            plt.scatter(ellipse_area_mean, label_os_mean[ind], color=plot_color, s=2+400*current_n/61, alpha=0.8-label_os_sem[ind])
        elif classmarkers[ind] == 3:
            plot_color=[0.8,0.2,0.8]
            plt.subplot(3,3,1)
            plt.scatter(np.nanmean(label_prefSFs), label_os_mean[ind], color=plot_color, s=2+400*current_n/61, alpha=0.8-label_os_sem[ind])
            plt.subplot(3,3,4)
            plt.scatter(ellipse_area_mean, label_os_mean[ind], color=plot_color, s=2+400*current_n/61, alpha=0.8-label_os_sem[ind])

        # define plot color by cell sign
        if signmarkers[ind] == 1:
            plot_color=[0.8,0.1,0.1]
            plt.subplot(3,3,2)
            plt.scatter(np.nanmean(label_prefSFs), label_os_mean[ind], color=plot_color, s=2+400*current_n/61, alpha=0.8-label_os_sem[ind])
        elif signmarkers[ind] == -1:
            plot_color=[0.1,0.1,0.8]
            plt.subplot(3,3,2)
            plt.scatter(np.nanmean(label_prefSFs), label_os_mean[ind], color=plot_color, s=2+400*current_n/61, alpha=0.8-label_os_sem[ind])

# axis organization
plt.subplot(3,3,1)
plt.ylim([1,2.25])
plt.ylabel('OS index (L/S)')
plt.xlabel('Pref SF (cycles/deg)')
plt.subplot(3,3,2)
# plt.ylim([1,2.5])
plt.ylim([1,2.25])
plt.xlabel('Pref SF (cycles/deg)')

plt.subplot(3,3,4)
plt.ylim([1,2.25])
plt.xlim([50,450])
plt.ylabel('OS index (L/S)')
plt.xlabel('Ellipse model area')

# spectrum by ON/OFF superclass w/ Nbytype>thresh_n across colors
plt.subplot(3,3,5)
plt.loglog(sf_vec[:15], np.nanmean(mean_label_spectra[:15,0,np.logical_and(signmarkers==1, Nbytype>thresh_n)],axis=(1)), color=[0.8,0.1,0.1], linewidth=2)
plt.loglog(sf_vec[:15], np.nanmean(mean_label_spectra[:15,0,np.logical_and(signmarkers==-1, Nbytype>thresh_n)],axis=(1)), color=[0.1,0.1,0.8], linewidth=2)
plt.ylim([0.003,0.22])
plt.ylabel('Log(Norm. Power)')
plt.xlabel('Log(Spatial Freq.)')
plt.text(0.02,0.01,'Blue')
plt.subplot(3,3,6)
plt.loglog(sf_vec[:15], np.nanmean(mean_label_spectra[:15,1,np.logical_and(signmarkers==1, Nbytype>thresh_n)],axis=(1)), color=[0.8,0.1,0.1], linewidth=2)
plt.loglog(sf_vec[:15], np.nanmean(mean_label_spectra[:15,1,np.logical_and(signmarkers==-1, Nbytype>thresh_n)],axis=(1)), color=[0.1,0.1,0.8], linewidth=2)
plt.ylim([0.003,0.22])
plt.xlabel('Log(Spatial Freq.)')
plt.text(0.02,0.01,'UV')

# spectrum by FF/FB/Lat superclass w/ Nbytype>thresh_n across colors
plt.subplot(3,3,7)
plt.loglog(sf_vec[:15], np.nanmean(mean_label_spectra[:15,:,np.logical_and(classmarkers==1,Nbytype>thresh_n)],axis=(1,2)), color=[0,0,0], linewidth=2)
plt.loglog(sf_vec[:15], np.nanmean(mean_label_spectra[:15,:,np.logical_and(classmarkers==1,Nbytype>thresh_n)],axis=(1)), color=[0,0,0], linewidth=0.5, alpha=0.3)
plt.ylim([0.003,0.22])
plt.ylabel('Log(Norm. Power)')
plt.xlabel('Log(Spatial Freq.)')

plt.subplot(3,3,8)
plt.loglog(sf_vec[:15], np.nanmean(mean_label_spectra[:15,:,np.logical_and(classmarkers==2,Nbytype>thresh_n)],axis=(1,2)), color=[0.2,0.8,0.2], linewidth=2)
plt.loglog(sf_vec[:15], np.nanmean(mean_label_spectra[:15,:,np.logical_and(classmarkers==2,Nbytype>thresh_n)],axis=(1)), color=[0.2,0.8,0.2], linewidth=0.5, alpha=0.5)
plt.ylim([0.003,0.22])
plt.xlabel('Log(Spatial Freq.)')

plt.subplot(3,3,9)
plt.loglog(sf_vec[:15], np.nanmean(mean_label_spectra[:15,:,np.logical_and(classmarkers==3,Nbytype>thresh_n)],axis=(1,2)), color=[0.8,0.2,0.8], linewidth=2)
plt.loglog(sf_vec[:15], np.nanmean(mean_label_spectra[:15,:,np.logical_and(classmarkers==3,Nbytype>thresh_n)],axis=(1)), color=[0.8,0.2,0.8], linewidth=0.5, alpha=0.4)
plt.ylim([0.003,0.22])
plt.xlabel('Log(Spatial Freq.)')

# superclass histogram calculations
ff_hist = np.histogram(areas_bytype[np.logical_and(classmarkers==1,Nbytype>thresh_n)], bins=10, range=(50,450))
ff_norm = ff_hist[0]
fb_hist = np.histogram(areas_bytype[np.logical_and(classmarkers==2,Nbytype>thresh_n)], bins=10, range=(50,450))
fb_norm = fb_hist[0]
lat_hist = np.histogram(areas_bytype[np.logical_and(classmarkers==3,Nbytype>thresh_n)], bins=10, range=(50,450))
lat_norm = lat_hist[0]
plt.subplot(3,3,3)
plt.plot(ff_hist[1][1:], ff_norm, color=[0,0,0]);
plt.plot(ff_hist[1][1:], fb_norm, color=[0.2,0.8,0.2]);
plt.plot(ff_hist[1][1:], lat_norm, color=[0.8,0.2,0.8]);
plt.ylabel('# Cell Types')
plt.xlabel('RF Center Area (sq. deg)')

np.sum(ff_hist[0])+np.sum(fb_hist[0])+np.sum(lat_hist[0])
# save it
fname = '/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/label summaries/OS_spectra_bytype.pdf'
spacefig.savefig(fname, format='pdf', orientation='landscape')

#%% generate a temporally smoothed version of STRFs for improving DS simulation results - only needs to be run once, prior to running DS simulations (in a different script)

def boxcar_smooth(y, fs, fs_new):
    # fs, fs_new in Hz
    box_pts = int(fs/fs_new)
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

# placeholder array for smoothed results - load with NaNs to recapitulate all_centered_STRFs structure
all_smoothed_STRFs = np.full(all_centered_STRFs.shape, np.nan, dtype='float')

for cell_ind in range(0,scraped_log.shape[0]):
    for c in range(0,2):
        # define current STRF to smooth
        current_strf = all_centered_STRFs[:,:,:,c,cell_ind]
        # skip this cell if the middle point is NaN (this occurs for cells with missing color data)
        if np.isnan(current_strf[39,39,50]):
            pass
        else:
            # placeholder arrays for temporally and spatially smoothed STRFs
            ts_strf = np.zeros(current_strf.shape)
            smoothed_strf = np.zeros(current_strf.shape)
            # spatial smooth on the original STRF
            for t in range(0,current_strf.shape[2]):
                current_image = current_strf[:,:,t]
                ts_strf[:,:,t] = ndimage.gaussian_filter(current_image, sigma=2, order=0, mode='constant', cval=0)
            # temporal smooth, use the spatially smoothed STRF as input
            for x in range(0,current_strf.shape[0]):
                for y in range(0,current_strf.shape[1]):
                    current_point = ts_strf[x,y,:]
                    smoothed_strf[x,y,:] = boxcar_smooth(current_point, fs=20, fs_new=4)
            # save smoothing result to array
            all_smoothed_STRFs[:,:,:,c,cell_ind] = smoothed_strf

# save the array so it can be moved to Oak for processing via Sherlock
np.save('/Volumes/TBD/Bruker/STRFs/compiled/all_smoothed_strfs.npy', all_smoothed_STRFs)

#%% load and analyze simulated DS data
# helper functions
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    v = np.asarray([rho, phi])
    return(v)
def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    p = np.asarray([x, y])
    return(p)

# load saved data
all_ds_sims = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/jointSTRF_DS_simulations_80_1_8_flip_smoothed.npy')
tp = 1
dt = 0.05

# create container variables
ds_index = np.full((scraped_log.shape[0]), np.nan, dtype='float')
peak_phase_diff = np.full((scraped_log.shape[0]), np.nan, dtype='float')
pop_vec_rel_len = np.full((scraped_log.shape[0]), np.nan, dtype='float')
os_by_phase_var = np.full((scraped_log.shape[0]), np.nan, dtype='float')

# loop over cells
for cell_ind in range(0,scraped_log.shape[0]):
    if not np.any(np.isnan(all_ds_sims[:,:,cell_ind])):
        # looks at the 6th cycle (steady-state of the convolved response) - qualitatively and quantitatively, cycles 4-7 are identical for a 1s temporal period and a 3 sec STRF
        if tp == 1:
            current_resp = all_ds_sims[int(6/(dt/tp)):int(7/(dt/tp)),:,cell_ind]
        elif tp == 2:
            # for slower stimuli (0.5 Hz), there are fewer cycles, so we want the middle one
            current_resp = all_ds_sims[int(3/(dt/tp)):int(4/(dt/tp)),:,cell_ind]
        elif tp == 0.5:
            current_resp = all_ds_sims[int(9/(dt/tp)):int(10/(dt/tp)),:,cell_ind]

        # DS INDEX BLOCK
        # find time and direction of largest value - do not need to consider absolute value here because the steady-state response is a sinusoid, with appx equal positive and negative lobes
        [peak_t,pd] = np.where(current_resp == np.max(current_resp, axis=(0,1)))
        peak_t = peak_t[0]
        pd = pd[0]
        # null direction is num_angles/2 steps away (180º)
        nd = int(np.mod(pd-current_resp.shape[1]/2,current_resp.shape[1]))
        # basic pd-nd responses at peak_t phase
        peak_phase_diff[cell_ind] = current_resp[peak_t,pd] - current_resp[peak_t,nd]
        # peak_phase_diff[cell_ind] = (current_resp[peak_t,pd] - np.max(current_resp[:,nd]))/(current_resp[peak_t,pd] + np.max(current_resp[:,nd]))
        # convert pd and nd waveforms into polar coordinates (18 deg or pi/10 radians per index for a 1 sec temporal period)(in general, 2*pi/(dt/tp) radians)
        rho_p = current_resp[peak_t,pd]
        phi_p = peak_t*(np.pi/10)/tp
        rho_n = np.max(current_resp[:,nd])
        phi_n = np.argmax(current_resp[:,nd])*(np.pi/10)/tp
        vdif = pol2cart(rho_p,phi_p)-pol2cart(rho_n,phi_n)
        # DS index is the length of the difference vector over the sum of the lengths of the two input vectors
        ds_index[cell_ind] = cart2pol(vdif[0],vdif[1])[0]/(rho_p+rho_n)

        # POPULATION VECTOR BLOCK
        # find rho and phi for each theta (representing stimulus angle)
        all_phi = np.argmax(current_resp,axis=0)*(np.pi/10)/tp
        all_rho = np.max(current_resp,axis=0)
        # the angle of each response vector is given by phi+theta
        all_theta = np.deg2rad(np.arange(0,360,360/current_resp.shape[1]))
        vec_ends = pol2cart(all_rho,all_phi+all_theta)
        # sum vector coordinates in cartesian space
        pop_end = np.sum(vec_ends,axis=1)
        # convert population vector back to polar coords for length, express it as a fraction of the sum of all vector lengths (this is the direction-general form of the specific equation for DS)
        pop_vec_rel_len[cell_ind] = cart2pol(pop_end[0],pop_end[1])[0]/np.sum(all_rho)
        # compute the circular variance of phi (response phase offset) across stimulus directions - this value is proportional to orientation tuning strength
        os_by_phase_var[cell_ind] = stats.circvar(all_phi, high=2*np.pi, low=0)

# save data arrays
np.save('/Volumes/TBD/Bruker/STRFs/compiled/all_ds_index.npy', ds_index)
np.save('/Volumes/TBD/Bruker/STRFs/compiled/all_peak_differences.npy', peak_phase_diff)
np.save('/Volumes/TBD/Bruker/STRFs/compiled/all_ds_popvec_relative_lengths.npy', pop_vec_rel_len)
np.save('/Volumes/TBD/Bruker/STRFs/compiled/all_os_index_by_phase_variance.npy', os_by_phase_var)

#%% DS simulation summary plot
thresh_n = 2
area_thresh = 105

utype_indices = np.unique(scraped_log[:,14]) != ''
unique_types = np.unique(scraped_log[:,14])[utype_indices]

# container arrays
label_dsi_mean = np.zeros((len(unique_types)))
label_pkdiff_mean = np.zeros((len(unique_types)))
label_pvec_len_mean = np.zeros((len(unique_types)))
label_osvar_mean = np.zeros((len(unique_types)))

for ind in range(0,len(unique_types)):
# for ind in range(50,57):
    current_label = unique_types[ind]
    current_cells = scraped_log[scraped_log[:,14] == current_label,1]
    current_n = len(current_cells)
    # grab metrics for current cell type
    label_areas = ellipse_areas[scraped_log[:,14] == current_label]
    label_dsi = ds_index[scraped_log[:,14] == current_label]
    label_pkdiff = peak_phase_diff[scraped_log[:,14] == current_label]
    label_pvec_len = pop_vec_rel_len[scraped_log[:,14] == current_label]
    label_osvar = os_by_phase_var[scraped_log[:,14] == current_label]
    # nan-out subthreshold areas
    label_dsi[label_areas < area_thresh] = np.nan
    label_pkdiff[label_areas < area_thresh] = np.nan
    label_pvec_len[label_areas < area_thresh] = np.nan
    # remove NaNs
    label_dsi = label_dsi[~np.isnan(label_dsi)]
    label_pkdiff = label_pkdiff[~np.isnan(label_pkdiff)]
    label_pvec_len = label_pvec_len[~np.isnan(label_pvec_len)]

    if len(np.unique(label_dsi)) > thresh_n:
        # add mean metrics to array for sorting
        label_dsi_mean[ind] = np.nanmedian(label_dsi)
        label_pkdiff_mean[ind] = np.nanmedian(label_pkdiff)
        label_pvec_len_mean[ind] = np.nanmedian(label_pvec_len)
        label_osvar_mean[ind] = np.nanmedian(label_osvar)

# remove thresholded cell types
thresh_types = unique_types[np.nonzero(label_dsi_mean)]
label_dsi_mean = label_dsi_mean[np.nonzero(label_dsi_mean)]
label_pkdiff_mean = label_pkdiff_mean[np.nonzero(label_pkdiff_mean)]
label_pvec_len_mean = label_pvec_len_mean[np.nonzero(label_pvec_len_mean)]
label_osvar_mean = label_osvar_mean[np.nonzero(label_osvar_mean)]

#%% plotting
dsfig=plt.figure(figsize=(15, 12))
plt.suptitle('DS Simulation Summary by Type (N>=' + str(thresh_n+1) + ')',fontsize='xx-large', fontweight='bold')

plt.subplot(411)
sorted_inds = np.argsort(label_dsi_mean)
for n in range(0, len(sorted_inds)):
    current_label = thresh_types[sorted_inds[n]]
    label_areas = ellipse_areas[scraped_log[:,14] == current_label]
    label_dsi = ds_index[scraped_log[:,14] == current_label]
    label_dsi[label_areas < area_thresh] = np.nan
    label_dsi = label_dsi[~np.isnan(label_dsi)]
    # if len(label_dsi) >= 6:
    #     plt.boxplot(label_dsi, positions=[n], widths=0.8, showfliers=False, showcaps=False, whis=[0,100], labels={''})
    # else:
    #     plt.plot([n, n], [np.max(label_dsi), np.min(label_dsi)], color='k', linewidth=0.75)
    plt.scatter(np.full(len(label_dsi),n),label_dsi,color=[0,0,0],s=8,alpha=0.3)
    plt.text(n, -0.2, current_label, rotation='vertical')
    plt.text(n, -0.4, str(len(label_dsi)))
plt.scatter(np.arange(0,len(thresh_types),1), label_dsi_mean[sorted_inds], color=[0,0,0], s=30)
plt.ylim([-0.25,1.05])
plt.ylabel('DSI (vector dif.)')

plt.subplot(412)
sorted_inds = np.argsort(label_pkdiff_mean)
for n in range(0, len(sorted_inds)):
    current_label = thresh_types[sorted_inds[n]]
    label_areas = ellipse_areas[scraped_log[:,14] == current_label]
    label_pkdiff = peak_phase_diff[scraped_log[:,14] == current_label]
    label_pkdiff[label_areas < area_thresh] = np.nan
    label_pkdiff = label_pkdiff[~np.isnan(label_pkdiff)]
    plt.scatter(np.full(len(label_pkdiff),n),label_pkdiff,color=[0,0,0],s=8,alpha=0.3)
    plt.text(n, -0.2, current_label, rotation='vertical')
    plt.text(n, -0.4, str(len(label_pkdiff)))
plt.scatter(np.arange(0,len(thresh_types),1), label_pkdiff_mean[sorted_inds], color=[0,0,0], s=30)
# plt.ylim([-0.25,0.9])
plt.ylabel('DSI (amplitdue dif.)')

plt.subplot(413)
sorted_inds = np.argsort(label_pvec_len_mean)
for n in range(0, len(sorted_inds)):
    current_label = thresh_types[sorted_inds[n]]
    label_areas = ellipse_areas[scraped_log[:,14] == current_label]
    label_pvec_len = pop_vec_rel_len[scraped_log[:,14] == current_label]
    label_pvec_len[label_areas < area_thresh] = np.nan
    label_pvec_len = label_pvec_len[~np.isnan(label_pvec_len)]
    # if len(label_pvec_len) >= 6:
    #     plt.boxplot(label_pvec_len, positions=[n], widths=0.8, showfliers=False, showcaps=False, whis=[0,100], labels={''})
    # else:
    #     plt.plot([n, n], [np.max(label_pvec_len), np.min(label_pvec_len)], color='k', linewidth=0.75)
    plt.scatter(np.full(len(label_pvec_len),n),label_pvec_len,color=[0,0,0],s=8,alpha=0.3)
    plt.text(n, -0.2, current_label, rotation='vertical')
    plt.text(n, -0.4, str(len(label_pvec_len)))
plt.scatter(np.arange(0,len(thresh_types),1), label_pvec_len_mean[sorted_inds], color=[0,0,0], s=30)
plt.ylim([-0.25,0.9])
plt.ylabel('Population vector rel. length')

plt.subplot(4,3,10)
plt.scatter(label_dsi_mean, label_pvec_len_mean, color=[0.2,0.2,0.2], s=20)
plt.scatter(label_dsi_mean[thresh_types == 'T4'], label_pvec_len_mean[thresh_types == 'T4'], color=[0.2,0.8,0.2], s=20)
plt.scatter(label_dsi_mean[thresh_types == 'T2'], label_pvec_len_mean[thresh_types == 'T2'], color=[0.8,0.2,0.8], s=20)
plt.scatter(label_dsi_mean[thresh_types == 'Y13'], label_pvec_len_mean[thresh_types == 'Y13'], color=[0.2,0.2,0.8], s=20)
plt.scatter(label_dsi_mean[thresh_types == 'Pm10'], label_pvec_len_mean[thresh_types == 'Pm10'], color=[0.8,0.2,0.2], s=20)
plt.xlabel('DSI (vector dif.)')
plt.ylabel('Population vector rel. length')

plt.subplot(4,3,11)
plt.scatter(label_dsi_mean, label_pkdiff_mean, color=[0.2,0.2,0.2], s=20)
plt.scatter(label_dsi_mean[thresh_types == 'T4'], label_pkdiff_mean[thresh_types == 'T4'], color=[0.2,0.8,0.2], s=20)
plt.scatter(label_dsi_mean[thresh_types == 'T2'], label_pkdiff_mean[thresh_types == 'T2'], color=[0.8,0.2,0.8], s=20)
plt.scatter(label_dsi_mean[thresh_types == 'Y13'], label_pkdiff_mean[thresh_types == 'Y13'], color=[0.2,0.2,0.8], s=20)
plt.scatter(label_dsi_mean[thresh_types == 'Pm10'], label_pkdiff_mean[thresh_types == 'Pm10'], color=[0.8,0.2,0.2], s=20)
plt.xlabel('DSI (vector dif.)')
plt.ylabel('DSI (amplitude dif.)')

plt.subplot(4,3,12)
plt.scatter(label_pkdiff_mean, label_pvec_len_mean, color=[0.2,0.2,0.2], s=20)
plt.scatter(label_pkdiff_mean[thresh_types == 'T4'], label_pvec_len_mean[thresh_types == 'T4'], color=[0.2,0.8,0.2], s=20)
plt.scatter(label_pkdiff_mean[thresh_types == 'T2'], label_pvec_len_mean[thresh_types == 'T2'], color=[0.8,0.2,0.8], s=20)
plt.scatter(label_pkdiff_mean[thresh_types == 'Y13'], label_pvec_len_mean[thresh_types == 'Y13'], color=[0.2,0.2,0.8], s=20)
plt.scatter(label_pkdiff_mean[thresh_types == 'Pm10'], label_pvec_len_mean[thresh_types == 'Pm10'], color=[0.8,0.2,0.2], s=20)
plt.xlabel('DSI (amplitdue dif.)')
plt.ylabel('Population vector rel. length')

# save it
fname = '/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/label summaries/DS_summary.pdf'
dsfig.savefig(fname, format='pdf', orientation='portrait')

#%% plot DS example

dsexfig = plt.figure(figsize=(15, 12))
plt.suptitle(scraped_log[cell_ind,1])

plt.subplot(221)
for n in range(0,len(all_rho)):
    v = pol2cart(all_rho[n],all_phi[n]+all_theta[n])
    plt.plot([0,v[0]],[0,v[1]],'k-')
    plt.plot([0,pop_end[0]],[0,pop_end[1]],'r-')
    plt.axis('equal')
    plt.axis('square')
    plt.title('Response Population Vector')
plt.text(-800000,800000,'Rel. L = ' + str(np.round(pop_vec_rel_len[cell_ind,c],2)))
pop_vec_rel_len.shape
plt.subplot(222)
if tp == 1:
    plt.plot(all_ds_sims[int(6/(dt/tp)):int(7/(dt/tp))+1,:,0,cell_ind])
elif tp == 0.5:
    plt.plot(all_ds_sims[int(9/(dt/tp)):int(10/(dt/tp))+1,:,0,cell_ind])
elif tp == 2:
    plt.plot(all_ds_sims[int(3/(dt/tp)):int(4/(dt/tp))+1,:,0,cell_ind])
plt.title('Steady-state Single Cycle Response')

plt.subplot(223)
plt.plot([0,pol2cart(rho_p,phi_p)[0]],[0,pol2cart(rho_p,phi_p)[1]],'k-')
plt.plot([0,pol2cart(rho_n,phi_n)[0]],[0,pol2cart(rho_n,phi_n)[1]],'r-')
plt.plot([0,vdif[0]],[0,vdif[1]],'b-')
plt.text(400000,0,'DSI = ' + str(np.round(ds_index[cell_ind,c],2)))
plt.axis('equal')
plt.axis('square')
plt.title('PD, ND, and PD-ND Vectors')

plt.subplot(224)
# plt.plot(all_ds_sims[int(3/(dt/tp)):int(4/(dt/tp))+1,pd,0,cell_ind],'k-')
if tp == 1:
    plt.plot(all_ds_sims[int(6/(dt/tp)):int(7/(dt/tp))+1,pd,0,cell_ind],'k-')
    plt.plot(all_ds_sims[int(6/(dt/tp)):int(7/(dt/tp))+1,nd,0,cell_ind],'r-')
elif tp == 2:
    plt.plot(all_ds_sims[int(3/(dt/tp)):int(4/(dt/tp))+1,pd,0,cell_ind],'k-')
    plt.plot(all_ds_sims[int(3/(dt/tp)):int(4/(dt/tp))+1,nd,0,cell_ind],'r-')
elif tp == 0.5:
    plt.plot(all_ds_sims[int(9/(dt/tp)):int(10/(dt/tp))+1,pd,0,cell_ind],'k-')
    plt.plot(all_ds_sims[int(9/(dt/tp)):int(10/(dt/tp))+1,nd,0,cell_ind],'r-')

plt.title('PD and ND Single Cycle Responses')

#%% save it
fname = '/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/label summaries/DS_example_Mi1.pdf'
dsexfig.savefig(fname, format='pdf', orientation='landscape')

#%% visual feature covariance matrix - modified from the full version in ME_build_correlate, which must be run first in order to generate the table that is loaded here! This version uses all cell types @ N>=3, rather than only those cell types that exist in the transcriptome dataset.

# load saved data
highN_table = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/data table/high_N_table.npy')
cell_type_list = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/data table/high_N_cell_type_list.npy')
feat_list = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/data table/feature_list.npy')
feat_list
# identify unique cell types
utypes = np.unique(cell_type_list)

# container table for mean data by cell type and nan-removed raw data
mean_table = np.full((len(utypes),highN_table.shape[1]), np.nan, dtype='float')
median_table = np.full((len(utypes),highN_table.shape[1]), np.nan, dtype='float')
nl_highN_table = np.zeros((1,highN_table.shape[1]))

# loop over unique types to populate mean table
for type in range(0,len(utypes)):
    current_label = utypes[type]
    type_data = highN_table[cell_type_list == current_label,:]
    mean_table[type,:] = np.nanmean(type_data, axis=0)
    median_table[type,:] = np.nanmedian(type_data, axis=0)

#%% plotting - full table
fullfeatfig = plt.figure(figsize=(15,15))
x = plt.imshow(-1*np.corrcoef(np.transpose(median_table)), cmap='PuOr', clim=(-1,1))
x.axes.get_yaxis().set_ticks(np.arange(0,89,1))
x.axes.get_yaxis().set_ticklabels(feat_list);
x.axes.get_xaxis().set_ticks(np.arange(0,89,1))
x.axes.get_xaxis().set_ticklabels(feat_list, rotation='vertical');
plt.colorbar(shrink=0.2, aspect=10)
fname = '/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all cell summaries/feature covariance matrix.pdf'
fullfeatfig.savefig(fname, format='pdf', orientation='landscape')

#%% plotting feature subsets
partialfeatfig = plt.figure(figsize=(18,18))
# spatial[47:63] vs temporal[5:35] spectra
plt.subplot(221)
tables_subset = np.append(np.append(np.append(median_table[:,5:35], median_table[:,47:63], axis=1), np.append(median_table[:,79:81], median_table[:,66:69], axis=1), axis=1), median_table[:,87:89], axis=1)
feat_subset = np.append(np.append(np.append(feat_list[5:35], feat_list[47:63]), np.append(feat_list[79:81], feat_list[66:69])), feat_list[87:89])
x = plt.imshow(-1*np.corrcoef(np.transpose(tables_subset)), cmap='PuOr', clim=(-1,1))
plt.plot([29.5,29.5], [-0.5,len(feat_subset)-0.5], '-', color=[0.2,0.7,0.3])
plt.plot([-0.5,len(feat_subset)-0.5], [29.5,29.5], '-', color=[0.2,0.7,0.3])
plt.plot([45.5,45.5], [-0.5,len(feat_subset)-0.5], '-', color=[0.2,0.7,0.3])
plt.plot([-0.5,len(feat_subset)-0.5], [45.5,45.5], '-', color=[0.2,0.7,0.3])
x.axes.get_yaxis().set_ticks(np.arange(0,len(feat_subset),1))
x.axes.get_yaxis().set_ticklabels(feat_subset);
x.axes.get_xaxis().set_ticks(np.arange(0,len(feat_subset),1))
x.axes.get_xaxis().set_ticklabels(feat_subset, rotation='vertical');

# AUCs[39:47:2], flicker MD, DC, DS[71:81], OS, area[66:71]
plt.subplot(222)
tables_subset = np.append(median_table[:,39:47:2], np.append(median_table[:,71:81], median_table[:,66:69], axis=1), axis=1)
feat_subset = np.append(feat_list[39:47:2], np.append(feat_list[71:81], feat_list[66:69]))
x = plt.imshow(-1*np.corrcoef(np.transpose(tables_subset)), cmap='PuOr', clim=(-1,1))
plt.plot([3.5,3.5], [-0.5,len(feat_subset)-0.5], '-', color=[0.2,0.7,0.3])
plt.plot([-0.5,len(feat_subset)-0.5], [3.5,3.5], '-', color=[0.2,0.7,0.3])
plt.plot([11.5,11.5], [-0.5,len(feat_subset)-0.5], '-', color=[0.2,0.7,0.3])
plt.plot([-0.5,len(feat_subset)-0.5], [11.5,11.5], '-', color=[0.2,0.7,0.3])
x.axes.get_yaxis().set_ticks(np.arange(0,len(feat_subset),1))
x.axes.get_yaxis().set_ticklabels(feat_subset);
x.axes.get_xaxis().set_ticks(np.arange(0,len(feat_subset),1))
x.axes.get_xaxis().set_ticklabels(feat_subset, rotation='vertical');


fname = '/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all cell summaries/feature covariance matrix subsets.pdf'
partialfeatfig.savefig(fname, format='pdf', orientation='landscape')

#%% visualize the distribution of RF center vs. surround z-scores, and plot the threshold used for SRF spatial statistic calculations (supplemental figure)
zstatsfig = plt.figure(figsize=(6, 4))
hist,bined = np.histogram(np.abs(all_centered_STRFs[39,39,50:,:,:]),bins=300,range=(0,4))
plt.semilogy(bined[1:],hist/np.sum(hist),color=[0.8,0.5,0.2])
hist,bined = np.histogram(np.abs(all_centered_STRFs[24,24,50:,:,:]),bins=300,range=(0,4))
plt.semilogy(bined[1:],hist/np.sum(hist),color=[0.2,0.5,0.5])
plt.semilogy([0.5,0.5],[0,0.2],':', color=[0.5,0.5,0.5])
plt.ylabel('log(probablility)')
plt.xlabel('z-score value')
plt.title('P(z) for RF centers and representative surrounds')

fname = '/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all cell summaries/z-score distributions center vs surroun.pdf'
zstatsfig.savefig(fname, format='pdf', orientation='landscape')

#%% plot SRF and TRF for four Tm3s from the same fly, along with their PC space pairwise distances (supplemental figure)

# load PC weights and center TRFs
pc_weights = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all_PC_weights.npy')
cent_TRFs = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all_center_TRFs.npy')

# specify cells to plot
cells = np.asarray(['ME8046-0298', 'ME8046-0285', 'ME8046-0279', 'ME8046-0296'])
cell_inds = np.zeros(len(cells), dtype='int')

# pull full log indices of the four cells
for n in range(0,len(cells)):
    cell_inds[n] = int(np.where(scraped_log[:,1] == cells[n])[0][0])

# define plot colors
plotcolors = np.asarray([[1,0,0],[0,0,1],[0,1,0],[1,1,0]])

# plot section
varsupp = plt.figure(figsize=(16, 12));
for n in range(0,len(cells)):
    plt.subplot(3,4,n+1)
    plt.imshow(all_uncetered_cell_SRFs[:,:,0,cell_inds[n]], origin='lower', cmap='PiYG',clim=[-1,1])
    # plt.plot(all_flick_resp[2,:,cell_inds[n]].T,'k-')
    # plt.ylim(-1,2)
    plt.subplot(3,4,n+5)
    current_TRFs = cent_TRFs[:,:,cell_inds[n]]
    plt.plot([0,60],[0,0],'k:')
    plt.plot(boxcar_smooth(current_TRFs[:,0], fs=20, fs_new=4), '-', color=plotcolors[n,:])
    plt.plot(boxcar_smooth(current_TRFs[:,1], fs=20, fs_new=4), '--', color=plotcolors[n,:])
    plt.ylim(-0.7,1.3)
    plt.subplot(3,4,n+9)
    for m in range(0,len(cells)):
        if not m == n:
            plt.plot(np.linalg.norm(pc_weights[cell_inds[n]]-pc_weights[cell_inds[m]]), 'o', color=plotcolors[m,:])
    plt.ylim(0,60)

# save the figure
fname = '/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/label summaries/Tm3 within-fly variability.pdf'
varsupp.savefig(fname, format='pdf', orientation='landscape')


#%% functional proofreading blocks - plot individual STRFs for a specified cell type, along with a metric of interest (here, DSI, but this can be changed to any cell-wise metric)
exstrffig = plt.figure(figsize=(18,12))
label = 'Mi13'

all_blue_resp = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all_blue_rois.npy')
label_STRFs = all_centered_STRFs[:,:,:,:,scraped_log[:,14] == label]
label_n = label_STRFs.shape[4]

# label_STRFs = all_centered_STRFs[:,:,:,:,scraped_log[:,14] == label]
# label_STRFs = all_smoothed_STRFs[:,:,:,:,scraped_log[:,14] == label]
label_ds = ds_index[scraped_log[:,14] == label]
# label_ds = orient_selects[scraped_log[:,14] == label]
label_cells = scraped_log[scraped_log[:,14] == label,1]
# label_n = label_STRFs.shape[4]

for cell in range(0,label_n):
    exstrffig = plt.figure(figsize=(20, 5));
    plt.suptitle(label_cells[cell] + ', DS =' + str(label_ds[cell]))
    for n in range(0,10):
        plt.subplot(2,10,(n+1));
        # plt.imshow(label_STRFs[:,:,50+(n),cell], origin='lower', cmap='PiYG',clim=[-3,3]);

        plt.imshow(label_STRFs[20:60,20:60,50+(n),0,cell], origin='lower', cmap='PiYG',clim=[-3,3]);
        # plt.imshow((label_STRFs[20:60,20:60,50+(n),0,cell]+label_STRFs[20:60,20:60,50+(n),1,cell])/2, origin='lower', cmap='PiYG',clim=[-2,2]);
        plt.title(str(np.round(3-((50+(n))/(20)),2)))
        plt.subplot(2,10,(n+11));
        plt.imshow(label_STRFs[20:60,20:60,50+(n),1,cell], origin='lower', cmap='PiYG',clim=[-3,3]);
        plt.title(str(np.round(3-((50+(n))/(20)),2)))
    # save the figure
    fname = '/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/label summaries/' + label + '_' + str(cell) + '_STRF_examples.pdf'
    exstrffig.savefig(fname, format='pdf', orientation='landscape')



#%%
exsrffig = plt.figure(figsize=(18,8))
label = 'Dm2'
# label_SRFs = all_uncetered_cell_SRFs[:,:,:,scraped_log[:,14] == label]
# label_SRFs = mean_peak_SRFs[:,:,:,scraped_log[:,14] == label]
label_SRFs = all_cell_SRFs[:,:,:,scraped_log[:,14] == label]
label_cells = scraped_log[scraped_log[:,14] == label,1]
label_n = label_SRFs.shape[3]

for cell in range(0,label_n):
    plt.subplot(2,label_n,cell+1)
    plt.title(label_cells[cell])
    plt.imshow(label_SRFs[:,:,0,cell], origin='lower', cmap='PiYG',clim=[-1,1]);
    plt.subplot(2,label_n,label_n+cell+1)
    plt.imshow(label_SRFs[:,:,1,cell], origin='lower', cmap='PiYG',clim=[-1,1]);


# save the figure
fname = '/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/label summaries/' + label + '_SRF_examples.pdf'
exsrffig.savefig(fname, format='pdf', orientation='landscape')

#%%
meansrffig = plt.figure(figsize=(18,8))
label = 'Dm13'
label_SRFs = mean_peak_SRFs[:,:,:,scraped_log[:,14] == label]
srf_mean = (np.nanmean(label_SRFs[20:60,20:60,0,:],axis=2)+np.nanmean(label_SRFs[20:60,20:60,1,:],axis=2)) /np.nanmax(np.abs((np.nanmean(label_SRFs[20:60,20:60,0,:],axis=2)+np.nanmean(label_SRFs[20:60,20:60,1,:],axis=2))))
blue_mean = np.nanmean(label_SRFs[20:60,20:60,0,:],axis=2)/np.nanmax(np.abs(np.nanmean(label_SRFs[20:60,20:60,0,:],axis=2)))
uv_mean = np.nanmean(label_SRFs[20:60,20:60,1,:],axis=2)/np.nanmax(np.abs(np.nanmean(label_SRFs[20:60,20:60,1,:],axis=2)))

plt.suptitle(label + " mean thresholded SRF")
plt.subplot(1,3,1)
plt.title('Blue')
plt.imshow(blue_mean, origin='lower', cmap='PiYG',clim=[-1,1]);
plt.subplot(1,3,2)
plt.title('UV')
plt.imshow(uv_mean, origin='lower', cmap='PiYG',clim=[-1,1]);
plt.subplot(1,3,3)
plt.title('Blue+UV')
plt.imshow(srf_mean, origin='lower', cmap='PiYG',clim=[-1,1]);


# save the figure
fname = '/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/label summaries/' + label + '_mean_SRF.pdf'
meansrffig.savefig(fname, format='pdf', orientation='landscape')

#%%

extrffig = plt.figure(figsize=(18,8))
label = 'Mi14'
label_TRFs = all_cent_TRFs[:,:,scraped_log[:,14] == label]
label_cells = scraped_log[scraped_log[:,14] == label,1]
label_n = label_TRFs.shape[2]

for cell in range(0,label_n):
    plt.subplot(2,label_n,cell+1)
    plt.title(label_cells[cell])
    plt.ylim(-3,3)
    plt.plot([0,60],[0,0],'k:')
    plt.plot(label_TRFs[:,0,cell], 'b-');
    plt.subplot(2,label_n,label_n+cell+1)
    plt.plot([0,60],[0,0],'k:')
    plt.plot(label_TRFs[:,1,cell], 'r-');
    plt.ylim(-3,3)

# save the figure
fname = '/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/label summaries/' + label + '_TRF_examples.pdf'
extrffig.savefig(fname, format='pdf', orientation='landscape')
