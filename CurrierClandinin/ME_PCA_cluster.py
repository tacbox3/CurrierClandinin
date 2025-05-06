# This script performs a principal components analysis on the responses of all ROIs in the dataset, then project individual ROIs into the resulting PC space. PC space coordinates for each ROI are saved as a .npy file.

from visanalysis.analysis import imaging_data, shared_analysis
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import cv2
import xlrd
import warnings
from pathlib import Path
from datetime import datetime
from scipy import ndimage
from sklearn.decomposition import PCA

book = xlrd.open_workbook('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/ME0708_proofed_culled_log.xls')
sheet = book.sheet_by_name('Sheet1')
data = [[sheet.cell_value(r, c) for c in range(sheet.ncols)] for r in range(sheet.nrows)]
full_log = np.asarray(data)

# this version of the log is already scraped to remove un-tagged and missing data ROIs; this is just removing the title and footnote rows
H5_bool = full_log[:,0] == 'o'
scraped_log = full_log[H5_bool,:]

# repopulate date strings from excel sequential day format
for roi_ind in range(0, np.shape(scraped_log)[0]):
    excel_date = int(float(scraped_log[roi_ind,2]))
    dt = datetime.fromordinal(datetime(1900, 1, 1).toordinal() + excel_date - 2)
    scraped_log[roi_ind,2] = str(dt)[0:10]

# open un-flattened culled response arrays
all_centered_STRFs = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/culled_centered_STRFs.npy')
all_rotated_STRFs = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/culled_rotated_STRFs.npy')
all_flick_resp = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/culled_flick_responses.npy')

# load pre-flattened and culled response matrices
culled_small_flat = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/culled_flattened_small_responses.npy')
culled_short_flat = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/culled_flattened_short_small_responses.npy')
culled_flat_means = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/culled_flattened_small_label_means.npy')

# flicker time vector and frequencies
flick_resamp_t = np.arange(0, 14-0.8, 1/5)
flick_freqs = [0.1,0.5,1,2]


#%% PCA on spatiotemporally sub-sampled and rotated data (center 30º and last 1 sec of STRF)
pca = PCA(n_components=100)
pc_weights = pca.fit_transform(culled_short_flat);
slice_ind = 36000 # for reconstructing the input matrix; equals total number of samples in the STRF


#%% visualize first n_pcs eigenvectors
n_pcs = 6

strfpc_fig = plt.figure(figsize=(16,20))
# tp = np.arange(52,58,1)
tp = np.arange(1,20,2)

for pc in range(0,n_pcs):
    # define the STRF and flicker for the PC
    pc_STRF = np.reshape(pca.components_[pc][:slice_ind], all_centered_STRFs[25:55,25:55,40:,:,:].shape[0:4])
    pc_flick = np.reshape(pca.components_[pc][slice_ind:], all_flick_resp.shape[0:2])
    # plot blue and UV STRF components
    for n in range(0,10):
        plt.subplot(n_pcs+n_pcs,11,((pc)*22)+(n+1));
        x = plt.imshow(pc_STRF[:,:,int(tp[n]),0], origin='lower', cmap='PiYG',clim=[-0.06,0.06]);
        if n != 0:
            x.axes.get_yaxis().set_ticks([])
            x.axes.get_xaxis().set_ticks([])
        else:
            plt.ylabel('PC' + str(pc) + ' Blue')
            x.axes.get_yaxis().set_ticks([0,30])
            x.axes.get_yaxis().set_ticklabels(['25','55'])
            x.axes.get_xaxis().set_ticks([])
        if pc == 0:
            plt.title(str(np.round((-1+((tp[n])+1)/20),2))+' s')
        x = plt.subplot(n_pcs+n_pcs,11,((pc)*22)+(n+12));
        plt.imshow(pc_STRF[:,:,int(tp[n]),1], origin='lower', cmap='PiYG',clim=[-0.06,0.06]);
        if n != 0:
            x.axes.get_yaxis().set_ticks([])
            x.axes.get_xaxis().set_ticks([])
        else:
            plt.ylabel('PC' + str(pc) + ' UV')
            x.axes.get_yaxis().set_ticks([0,30])
            x.axes.get_yaxis().set_ticklabels(['25','55'])
            x.axes.get_xaxis().set_ticks([])
    x = plt.subplot(n_pcs+n_pcs,11,((pc)*22)+11);
    plt.axis('off')
    plt.colorbar(fraction=0.8, aspect=8, ticks=[-0.04,0,0.04], label='PC weight', shrink=1);

flickpc_fig = plt.figure(figsize=(15,4))
# plot flicker components
for pc in range(0,n_pcs):
    # define the STRF and flicker for the PC
    pc_STRF = np.reshape(pca.components_[pc][:slice_ind], all_centered_STRFs[25:55,25:55,40:,:,:].shape[0:4])
    pc_flick = np.reshape(pca.components_[pc][slice_ind:], all_flick_resp.shape[0:2])
    plt.subplot(1,n_pcs,pc+1);
    plt.title('PC' + str(pc))
    for n in range(0,4):
        x = plt.plot(flick_resamp_t,pc_flick[n,:]);
        plt.ylim(-.03,.03)
        x[0].axes.get_yaxis().set_ticks([])
        x[0].axes.get_xaxis().set_ticks([2,12])
        if pc != 0:
            x[0].axes.get_yaxis().set_ticks([])
            x[0].axes.get_xaxis().set_ticks([2,12])
        else:
            plt.ylabel('Flicker Weight')
            x[0].axes.get_yaxis().set_ticks([-0.02,0,0.02])
            x[0].axes.get_xaxis().set_ticks([2,12])
        plt.plot([0,14],[0,0],'k:')

#%% save as PDF
fname_strf = '/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all cell summaries/STRF principal components.pdf'
fname_flick = '/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all cell summaries/flicker principal components.pdf'
strfpc_fig.savefig(fname_strf, format='pdf', orientation='portrait')
flickpc_fig.savefig(fname_flick, format='pdf', orientation='landscape')

#%% save the PC weights for first n_pcs
np.save('/Volumes/TBD/Bruker/STRFs/compiled/culled_PC_weights.npy', pc_weights[:,0:n_pcs])
np.save('/Volumes/TBD/Bruker/STRFs/compiled/culled_full_PC_weights.npy', pc_weights)

#%% summarize the PCA
# PCs after the first 4 do not explain as much variance

pcasumfig = plt.figure(figsize=(8,6))
plt.subplot(2,2,1)
plt.plot(pca.singular_values_)
plt.ylabel('Eigenvalue')
plt.xlabel('PC #')
plt.ylim(0,500)
plt.subplot(2,2,2)
plt.plot(pca.singular_values_[0:11])
plt.ylim(0,500)
plt.xlabel('PC #')

plt.subplot(2,2,3)
x = plt.plot(np.append(0,np.cumsum(pca.explained_variance_ratio_))*100)
plt.xlabel('# of PCs')
plt.ylabel('Explained Variance (''%'' of total)')
plt.ylim(0,60)
plt.subplot(2,2,4)
x = plt.plot(np.append(0,np.cumsum(pca.explained_variance_ratio_[0:10]))*100)
plt.xlabel('# of PCs')
plt.ylim(0,35)

# save it
fname = '/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all cell summaries/pca_summary.pdf'
pcasumfig.savefig(fname, format='pdf', orientation='landscape')

# %% COMPARE PAIRWISE DISTANCE METRICS (across cell types, within each cell type)
from scipy import stats

# pull list of unique cell type labels
utype_indices = np.unique(full_log[1:,14]) != ''
unique_types = np.unique(full_log[1:,14])[utype_indices]

# number of PCs to use in distance calc:
n_pcs = 100

# calculate and plot all pairwise distances for each cell type
distfig = plt.figure(figsize=(20,6))
plot_counter=0;
for n in range(0,len(unique_types)):
    label = unique_types[n]
    label_pc_coords = pc_weights[scraped_log[:,14] == label,0:n_pcs]
    if len(label_pc_coords) > 2:
        all_dist = []
        for main_cell in range(0,label_pc_coords.shape[0]):
            for second_cell in range(main_cell+1,label_pc_coords.shape[0]):
                all_dist = np.append(all_dist, np.linalg.norm(label_pc_coords[main_cell,:]-label_pc_coords[second_cell,:]))
        plt.violinplot(all_dist, positions=[plot_counter], widths=label_pc_coords.shape[0]/20, showextrema=False)
        plt.plot(plot_counter,np.mean(all_dist),'k.')
        plt.text(plot_counter,70,label, rotation='vertical')
        plot_counter = plot_counter+1
# calculate and plot all pairwise distances for all cells
all_dist = []
cross_dist = []
within_dist = []
for main_cell in range(0,pc_weights.shape[0]):
    for second_cell in range(main_cell+1,pc_weights.shape[0]):
        all_dist = np.append(all_dist, np.linalg.norm(pc_weights[main_cell,0:n_pcs]-pc_weights[second_cell,0:n_pcs]))
        if scraped_log[main_cell,14] != scraped_log[second_cell,14]:
            cross_dist = np.append(cross_dist, np.linalg.norm(pc_weights[main_cell,0:n_pcs]-pc_weights[second_cell,0:n_pcs]))
        else:
            within_dist = np.append(within_dist, np.linalg.norm(pc_weights[main_cell,0:n_pcs]-pc_weights[second_cell,0:n_pcs]))
plt.violinplot(all_dist, positions=[-3], widths=0.9, showextrema=False)
plt.plot(-3,np.median(all_dist),'k.')
plt.text(-2.5,70,'All pairs', rotation='vertical')
plt.violinplot(within_dist, positions=[-4.5], widths=0.9, showextrema=False)
plt.plot(-4.5,np.median(within_dist),'k.')
plt.text(-4,70,'Within-Type', rotation='vertical')
plt.violinplot(cross_dist, positions=[-6], widths=0.9, showextrema=False)
plt.plot(-6,np.median(cross_dist),'k.')
plt.text(-5.5,70,'Cross-Type', rotation='vertical')
plt.plot([-7,49],[np.median(cross_dist),np.median(cross_dist)],'k:')
plt.ylim([0, 100])
plt.xlim([-7, 49])
plt.title('Pairwise Euclidean Distances in PC space')
res = stats.mannwhitneyu(cross_dist,within_dist)
plt.text(-2,60,str(res.pvalue))


# save it
fname = '/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all cell summaries/pairwise_distance_(PCspace).pdf'
distfig.savefig(fname, format='pdf', orientation='landscape')
