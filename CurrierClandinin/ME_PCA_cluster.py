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


#%% PCA on spatially sub-sampled and rotated data (center 40º of STRF)
pca = PCA(n_components=100)
pc_weights = pca.fit_transform(culled_small_flat);
slice_ind = 192000 # for reconstructing the input matrix; equals total number of samples in the STRF

#%% PCA on spatiotemporally sub-sampled and rotated data (center 30º and last 1 sec of STRF)
pca = PCA(n_components=100)
pc_weights = pca.fit_transform(culled_short_flat);
slice_ind = 36000

#%% PCA on spatiotemporally sub-sampled and rotated label mean data (center 30º and last 1 sec of STRF, uses CELL TYPES as samples instead of individual cells)
pca = PCA(n_components=50)
pca.fit(culled_flat_means);
# after fitting on cell type means, project individual cells into PC space to get weights
pc_weights = pca.transform(culled_short_flat);
slice_ind = 36000

#%% visualize first n_pcs eigenvectors
n_pcs = 6

strfpc_fig = plt.figure(figsize=(16,20))
# tp = np.arange(52,58,1)
tp = np.arange(1,20,2)

for pc in range(0,n_pcs):
    # define the STRF and flicker for the PC
    # pc_STRF = np.reshape(pca.components_[pc][:slice_ind], all_centered_STRFs.shape[0:4])
    # pc_STRF = np.reshape(pca.components_[pc][:slice_ind], all_centered_STRFs[20:60,20:60,:,:,:].shape[0:4])
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
    # pc_STRF = np.reshape(pca.components_[pc][:slice_ind], all_centered_STRFs.shape[0:4])
    # pc_STRF = np.reshape(pca.components_[pc][:slice_ind], all_centered_STRFs[20:60,20:60,:,:,:].shape[0:4])
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

#%% Plot PC loadings by cell type
thresh_n = 2

# discover unique types in culled dataset
utype_indices = np.unique(scraped_log[:,14]) != ''
unique_types = np.unique(scraped_log[:,14])[utype_indices]

# initialize container for mean loadings
mean_weights = np.zeros((len(unique_types),n_pcs))

for ind in range(0,len(unique_types)):
    current_label = unique_types[ind]
    current_cells = scraped_log[scraped_log[:,14] == current_label,1]
    current_n = len(np.unique(current_cells))

    if current_n > thresh_n:
        # grab loadings for current cell type
        label_weights = pc_weights[scraped_log[:,14] == current_label,:n_pcs]
        # add mean weights to array
        mean_weights[ind,:] = np.mean(label_weights, axis=0)

# remove entries of cell types that do not meet threshold
thresh_types = unique_types[np.nonzero(mean_weights[:,0])[0]]
mean_weights = mean_weights[np.nonzero(mean_weights[:,0])[0]]

loadfig = plt.figure(figsize=(10, 15))
for pc in range(0,n_pcs):
    # sort types by mean loading
    sorted_inds = np.argsort(mean_weights[:,pc])
    sorted_types = thresh_types[sorted_inds]
    # determine normalization factor for current pc
    normfact = np.max(np.abs(pc_weights[:,pc]))
    # plot mean loadings
    plt.subplot(n_pcs,1,pc+1)
    plt.plot(mean_weights[sorted_inds,pc]/normfact,'k.')
    plt.plot([0,len(sorted_types)+1],[0,0],'k:')
    plt.ylim([-1.2,1.2])
    plt.xlim([-2,len(sorted_types)+2])
    # plot individual cells
    for ind in range(0,len(sorted_types)):
        current_label = sorted_types[ind]
        current_cells = scraped_log[scraped_log[:,14] == current_label,1]
        label_weights = pc_weights[scraped_log[:,14] == current_label,:n_pcs]
        plt.scatter(ind*np.ones(len(current_cells)), label_weights[:,pc]/normfact, color=[0.5,0.5,0.5], s=5)
        plt.text(ind,-1.1,current_label,rotation='vertical')


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


#%% write videos for first n_pcs

for pc in range(0,n_pcs):
    # re-extract PC STRFs
    pc_STRF = np.reshape(pca.components_[pc][:slice_ind], all_centered_STRFs[20:60,20:60,:,:,:].shape[0:4])
    pc_blue = pc_STRF[:,:,:,0]
    pc_uv = pc_STRF[:,:,:,1]
    # oversample z-scored STRF in x and y so video looks better
    big_strf=pc_blue.repeat(20,axis=0)
    bigger_strf=big_strf.repeat(20,axis=1)
    # oversample z-scored STRF in t so framerate can be reduced
    biggest_strf=bigger_strf.repeat(2,axis=2)
    # split into positive and negative patches
    pos_strf=np.power(np.where(biggest_strf>0,biggest_strf,0),1)
    neg_strf=np.power(np.where(biggest_strf<0,biggest_strf*-1,0),1)
    # convert z-scored STRF to -1 to 1 scale (units are standard deviations)
    low_lim = -0.04
    high_lim = 0.04
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
    video = cv2.VideoWriter('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all cell summaries/PC' + str(pc) + '_blue.mp4', cv2.VideoWriter_fourcc(*'mp4v'), float(fps), (p_new_strf.shape[1],p_new_strf.shape[0]))
    for frame_count in range(int(p_new_strf.shape[2]/3),int(p_new_strf.shape[2])):
        #this currently only plots the last second of an un-reveresed filter
        img = rgb_strf[:,:,frame_count,:]
        video.write(img)
    video.release()
    # repeat for UV component
    big_strf=pc_uv.repeat(20,axis=0)
    bigger_strf=big_strf.repeat(20,axis=1)
    biggest_strf=bigger_strf.repeat(2,axis=2)
    pos_strf=np.power(np.where(biggest_strf>0,biggest_strf,0),1)
    neg_strf=np.power(np.where(biggest_strf<0,biggest_strf*-1,0),1)
    low_lim = -0.04
    high_lim = 0.04
    p_new_strf = ((pos_strf - low_lim) * (2/(high_lim - low_lim))) - 1
    n_new_strf = ((neg_strf - low_lim) * (2/(high_lim - low_lim))) - 1
    p_new_strf=np.where(p_new_strf>1,1,p_new_strf)
    n_new_strf=np.where(n_new_strf>1,1,n_new_strf)
    rgb_strf=np.zeros((p_new_strf.shape+(3,)))
    rgb_strf[:,:,:,0]=1-(p_new_strf*1)-(n_new_strf*.3)
    rgb_strf[:,:,:,2]=1-(p_new_strf*1)-(n_new_strf*0)
    rgb_strf[:,:,:,1]=1-(p_new_strf*.3)-(n_new_strf*1)
    rgb_strf=np.where(rgb_strf>1,1,rgb_strf)
    rgb_strf = (rgb_strf*255).astype('uint8')
    fps = 10
    video = cv2.VideoWriter('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all cell summaries/PC' + str(pc) + '_uv.mp4', cv2.VideoWriter_fourcc(*'mp4v'), float(fps), (p_new_strf.shape[1],p_new_strf.shape[0]))
    for frame_count in range(int(p_new_strf.shape[2]/3),int(p_new_strf.shape[2])):
        #this currently only plots the last second of an un-reveresed filter
        img = rgb_strf[:,:,frame_count,:]
        video.write(img)
    video.release()

#%% Functional Clustering

# load pre-culled summary matrices
psd1D = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/culled_spatial_spectra.npy')
psdt = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/culled_temporal_spectra.npy')
psd1D_pow = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/culled_spatial_power.npy')
psdt_pow = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/culled_temporal_power.npy')
before_zc_auc = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/culled_before_last_lobe_AUC.npy')
after_zc_auc = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/culled_last_lobe_AUC.npy')
blue_nl_params = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/culled_blue_NL_fits.npy')[2:4,0,:] #indexing here pulls only slope and offset params from full fit
uv_nl_params = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/culled_uv_NL_fits.npy')[2:4,0,:]

# %% further indexing of culled summary metrics to only include ROIs where a sigmoidal fit was found for both blue and UV ()
sigfits_only = np.where(~np.isnan(uv_nl_params[0,:])&~np.isnan(blue_nl_params[0,:]))
blue_nl_params = blue_nl_params[:,sigfits_only]
uv_nl_params = uv_nl_params[:,sigfits_only]
psd1D = psd1D[:,:,sigfits_only]
psdt = psdt[:,:,sigfits_only]
psd1D_pow = psd1D_pow[:,sigfits_only]
psdt_pow = psdt_pow[:,sigfits_only]
before_zc_auc = before_zc_auc[:,sigfits_only]
after_zc_auc = after_zc_auc[:,sigfits_only]
scraped_log = np.squeeze(scraped_log[sigfits_only,:])


# %% build feature vectors
try:
    alll = np.where(~np.isnan(psd1D_pow[0,:]))
    psd1D_pow = psd1D_pow[:,alll]
    psdt_pow = psdt_pow[:,alll]
    before_zc_auc = before_zc_auc[:,alll]
    after_zc_auc = after_zc_auc[:,alll]
except:
    pass

feat_vecs = np.transpose(np.squeeze(np.mean(psd1D,axis=1)))
feat_vecs = np.append(feat_vecs,np.transpose(np.squeeze(np.mean(psdt,axis=1))),axis=1)
feat_vecs = np.append(feat_vecs,np.transpose(np.mean(psd1D_pow,axis=0))/np.amax(np.mean(psd1D_pow,axis=0)),axis=1)
feat_vecs = np.append(feat_vecs,np.transpose(np.mean(psdt_pow,axis=0))/np.amax(np.mean(psdt_pow,axis=0)),axis=1)
feat_vecs = np.append(feat_vecs,np.transpose(np.squeeze(before_zc_auc)),axis=1)
feat_vecs = np.append(feat_vecs,np.transpose(np.squeeze(after_zc_auc)),axis=1)
feat_vecs = np.append(feat_vecs,pc_weights[:,0:n_pcs]/np.tile(np.max(np.abs(pc_weights[:,0:n_pcs]),axis=(0)),(pc_weights.shape[0],1)),axis=1)


# try:
#     feat_vecs = np.append(feat_vecs,np.transpose(np.squeeze(blue_nl_params)),axis=1)
#     feat_vecs = np.append(feat_vecs,np.transpose(np.squeeze(uv_nl_params)),axis=1)
# except:
#     feat_vecs = feat_vecs


# %% Cluster UMAP embedded feature vectors with HDBSCAN method
import umap
from sklearn.cluster import HDBSCAN
reducer = umap.UMAP()
embedding = reducer.fit_transform(feat_vecs)
hdb = HDBSCAN(min_cluster_size=10,max_cluster_size=200)
clust_assignments = hdb.fit_predict(embedding)

# print labels of neurons in each cluster
for clust in range(0,np.amax(clust_assignments)+1):
    clust_labels = scraped_log[clust_assignments == clust,14]
    print('Cluster ' + str(clust) + ' contains: ' + str(np.sort(clust_labels)))
#%% plot clustered embedding
umap_fig = plt.figure()
for n in range(0,np.amax(clust_assignments)+1):
    plt.scatter(embedding[clust_assignments == n,0],embedding[clust_assignments == n,1])
# plt.scatter(embedding[clust_assignments == -1,0],embedding[clust_assignments == -1,1],color='k')
# plt.scatter(embedding[scraped_log[:,14] == 'TmY3',0],embedding[scraped_log[:,14] == 'TmY3',1],color='k')
# plt.scatter(embedding[scraped_log[:,14] == 'T2',0],embedding[scraped_log[:,14] == 'T2',1],color='k')
# plt.scatter(embedding[scraped_log[:,14] == 'Tm1',0],embedding[scraped_log[:,14] == 'Tm1',1],color='k')
# plt.scatter(embedding[scraped_log[:,14] == 'Tm2',0],embedding[scraped_log[:,14] == 'Tm2',1],color='k')
plt.scatter(embedding[scraped_log[:,14] == 'Tm4',0],embedding[scraped_log[:,14] == 'Tm4',1],color='k')
# plt.scatter(embedding[scraped_log[:,14] == 'Tm9',0],embedding[scraped_log[:,14] == 'Tm9',1],color='k')
# plt.scatter(embedding[scraped_log[:,14] == 'Tm20',0],embedding[scraped_log[:,14] == 'Tm20',1],color='k')
# plt.scatter(embedding[scraped_log[:,14] == 'Tm21',0],embedding[scraped_log[:,14] == 'Tm21',1],color='k')
plt.title('Clustered UMAP Embedding')

# %%
# save as PDF

fname = '/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/cluster summaries/UMAP_embedding.pdf'
umap_fig.savefig(fname, format='pdf', orientation='landscape')

#%% plot neurons of a given label in UMAP space by fly ID, and return IDs of matching ROIs for cross-validation.

label = 'Tm4'

plt.figure(figsize=(8, 6))
plt.scatter(embedding[:,0],embedding[:,1],color=[0.5,0.5,0.5],s=20)

label_roi_ids = scraped_log[scraped_log[:,14] == label,1]
label_fly_ids = np.zeros(len(label_roi_ids),dtype='int')
for n in range(0,len(label_roi_ids)):
     label_fly_ids[n] = label_roi_ids[n][3:6]

unique_flies = np.unique(label_fly_ids)
for cell in range(0,scraped_log.shape[0]):
    if scraped_log[cell,14] == label:
        # this plots all flies on the same axis
        # for fly in range(0,len(unique_flies)):
        #     if int(scraped_log[cell,1][3:6]) == unique_flies[fly]:
        #         plt.scatter(embedding[cell,0],embedding[cell,1],color=[fly/len(unique_flies),0,1-(fly/len(unique_flies))])
        #         print(scraped_log[cell,1])

        # this plots one fly at a time
        fly=4
        if int(scraped_log[cell,1][3:6]) == unique_flies[fly]:
            plt.scatter(embedding[cell,0],embedding[cell,1],color=[fly/len(unique_flies),0,1-(fly/len(unique_flies))],s=60)
            print(scraped_log[cell,1]+' (index = '+str(cell)+')')

# %% for a given label, plot euclidean distance from 0 (vector length) and cluster assignment for each neuron as a function of X, Y, and theta shift. This can tell me if neurons living near the edge of the screen, or neurons rotated in a particular way during alignment, are clustering separately. (This does not appear to be the case, lending support to the biological or fly-by-fly noise models). Still need to revisit labels and rule out by-fly SNR differences

label = 'Tm3'
culled_shifts = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/culled_shifts.npy')
label_shifts = culled_shifts[:,scraped_log[:,14] == label]
label_UMAP = embedding[scraped_log[:,14] == label,:]
label_norms = np.linalg.norm(label_UMAP,axis=1)
label_clusts = clust_assignments[scraped_log[:,14] == label]

plt.figure(figsize=(14, 8))
plt.suptitle('Does ' + label + ' cluster assignment vary with STRF center position/orientation?')
plt.subplot(231)
plt.scatter(np.abs(label_shifts[0,:]),label_norms)
plt.ylabel('Distance from UMAP Origin')
plt.xlim((-1,40))
plt.subplot(232)
plt.scatter(np.abs(label_shifts[1,:]),label_norms)
plt.xlim((-1,40))
plt.subplot(233)
plt.scatter(label_shifts[2,:],label_norms)
plt.subplot(234)
plt.scatter(np.abs(label_shifts[0,:]),label_clusts)
plt.ylabel('Cluster #')
plt.xlabel('Abs. X-shift (º)')
plt.xlim((-1,40))
plt.subplot(235)
plt.scatter(np.abs(label_shifts[1,:]),label_clusts)
plt.xlabel('Abs. Y-shift (º)')
plt.xlim((-1,40))
plt.subplot(236)
plt.scatter(label_shifts[2,:],label_clusts)
plt.xlabel('Rotation (º)')



#%% Plot summary for all clusters - not currently operational

for clust in range(-1,np.amax(clust_assignments)+1):
    # grab responses at indices where label matches cell type column
    label_STRFs = all_rotated_STRFs[:,:,:,:,clust_assignments == clust]
    label_TRFs = label_STRFs[40,40,:,:,:]
    label_flicks = all_flick_resp[:,:,clust_assignments == clust]
    n_cells = label_STRFs.shape[4]
    # find latest zero crossing of center for each TRF
    blue_SRFs = np.full((label_STRFs.shape[0],label_STRFs.shape[1],label_STRFs.shape[4]), np.NaN, dtype='float')
    uv_SRFs = np.full((label_STRFs.shape[0],label_STRFs.shape[1],label_STRFs.shape[4]), np.NaN, dtype='float')
    for cell in range(0,n_cells):
        if not np.isnan(np.mean(label_TRFs[:,0,:],axis=0))[cell]:
            blue_diff = np.diff(np.sign(label_TRFs[:,0,cell]))
            if np.nanmean(label_TRFs[37,0,:]) < 0:
                if np.any(np.where(blue_diff<0)):
                    blue_TRF_zcs = np.amax(np.where(np.where(blue_diff<0,blue_diff,0)))
                else:
                    blue_TRF_zcs = 34
            else:
                if np.any(np.where(blue_diff>0)):
                    blue_TRF_zcs = np.amax(np.where(np.where(blue_diff>0,blue_diff,0)))
                else:
                    blue_TRF_zcs = 34
        # perform same operation for uv SRF
        if not np.isnan(np.mean(label_TRFs[:,1,:],axis=0))[cell]:
            uv_diff = np.diff(np.sign(label_TRFs[:,1,cell]))
            if np.nanmean(label_TRFs[38,1,:]) < 0:
                if np.any(np.where(uv_diff<0)):
                    uv_TRF_zcs = np.amax(np.where(np.where(uv_diff<0,uv_diff,0)))
                else:
                    uv_TRF_zcs = 34
            else:
                if np.any(np.where(uv_diff>0)):
                    uv_TRF_zcs = np.amax(np.where(np.where(uv_diff>0,uv_diff,0)))
                else:
                    uv_TRF_zcs = 34

            # take SRF for period after latest zero crossing
            cell_blue_SRF = np.mean(label_STRFs[:,:,blue_TRF_zcs+1:40,0,cell],axis=2)
            blue_SRFs[:,:,cell] = cell_blue_SRF
            cell_uv_SRF = np.mean(label_STRFs[:,:,uv_TRF_zcs+1:40,1,cell],axis=2)
            uv_SRFs[:,:,cell] = cell_uv_SRF

    # calculate mean responses and std
    mean_STRF = np.nanmean(label_STRFs, axis=4)
    STRF_std = np.nanstd(label_STRFs, axis=4)
    mean_flick = np.nanmean(label_flicks, axis=2)
    flick_sem = np.nanstd(label_flicks, axis=2)/np.sqrt(n_cells)
    # calculate summary figures
    cent_TRF = mean_STRF[40,40,:,:]
    cent_TRF_sem = STRF_std[40,40,:,:]/np.sqrt(n_cells)
    blue_SRF = np.nanmean(blue_SRFs,axis=2)
    uv_SRF = np.nanmean(uv_SRFs,axis=2)

    # define scale factor for low n cell types
    if n_cells<4:
        sf = (1/((6-n_cells/1.5)))*2
    else:
        sf = 1
    sumfig = plt.figure(figsize=(15, 8))
    subfigs = sumfig.subfigures(3,1,height_ratios=[1.5,0.75,1.2])
    # blue STRF center
    axs0 = subfigs[0].subplots(2, 7)
    tp = np.arange(19,40,4)
    for n in range(0,6):
        x = plt.subplot(2,7,(n+1));
        plt.imshow(mean_STRF[20:60,20:60,int(tp[n]),0]*sf, origin='lower', cmap='PiYG',clim=[-1,1]);
        plt.title(str(np.round((-2+((tp[n]+1))/20),2))+' s')
        x.axes.get_xaxis().set_ticks([])
        if n != 0:
            x.axes.get_yaxis().set_ticks([])
        else:
            plt.ylabel('Blue')
            x.axes.get_yaxis().set_ticks([0,40])
            x.axes.get_yaxis().set_ticklabels(['20','60'])
    x = plt.subplot(2,7,7);
    plt.axis('off')
    plt.colorbar(fraction=0.8, aspect=8, ticks=[-1,0,1], label='Filter Amplitude', shrink=1);
    # UV STRF center
    for n in range(0,6):
        x = plt.subplot(2,7,(n+8));
        plt.imshow(mean_STRF[20:60,20:60,int(tp[n]),1]*sf, origin='lower', cmap='PiYG',clim=[-1,1]);
        x.axes.get_xaxis().set_ticks([0,40])
        x.axes.get_xaxis().set_ticklabels(['20','60'])
        if n != 0:
            x.axes.get_yaxis().set_ticks([])
        else:
            plt.ylabel('UV')
            x.axes.get_yaxis().set_ticks([0,40])
            x.axes.get_yaxis().set_ticklabels(['20','60'])
    x = plt.subplot(2,7,14);
    plt.axis('off')
    # flicker responses
    axs1 = subfigs[1].subplots(1, 4)
    flick_freqs = [0.1,0.5,1,2]
    for n in range(0,4):
        x = plt.subplot(1,4,(n+1));
        plt.plot(flick_resamp_t,mean_flick[n,:]*sf,'k')
        plt.plot(flick_resamp_t,(mean_flick[n,:]+flick_sem[n,:])*sf,'k',linewidth=0.5)
        plt.plot(flick_resamp_t,(mean_flick[n,:]-flick_sem[n,:])*sf,'k',linewidth=0.5)
        plt.title(str(flick_freqs[n])+' Hz')
        if n != 0:
            x.axes.get_yaxis().set_ticks([])
        else:
            if sf !=1:
                plt.ylabel('Scaled Flicker dF/F')
            else:
                plt.ylabel('Mean Flicker dF/F')
        x.axes.get_xaxis().set_ticks([2,7,12])
        x.axes.get_xaxis().set_ticklabels(['0','5','10'])
        plt.ylim([-0.5,1.25])
    # TRFs
    axs2 = subfigs[2].subplots(1,5)
    ax = plt.subplot(1,5,1)
    plt.plot(np.arange(-2,0,1/20),cent_TRF[:,0]*sf,color=[0.1,0.2,0.8])
    plt.plot(np.arange(-2,0,1/20),(cent_TRF[:,0]+cent_TRF_sem[:,0])*sf,color=[0.1,0.2,0.8],linewidth=0.5)
    plt.plot(np.arange(-2,0,1/20),(cent_TRF[:,0]-cent_TRF_sem[:,0])*sf,color=[0.1,0.2,0.8],linewidth=0.5)
    plt.plot([-2,0],[0,0],'k:')
    plt.title('Blue')
    plt.ylim([-1.25,1.25])
    ax.axes.get_xaxis().set_ticks([-2,-1,0])
    ax.axes.get_yaxis().set_ticks([-1,0,1])
    ax = plt.subplot(1,5,2)
    plt.plot(np.arange(-2,0,1/20),cent_TRF[:,1]*sf,color=[1,0,1])
    plt.plot(np.arange(-2,0,1/20),(cent_TRF[:,1]+cent_TRF_sem[:,1])*sf,color=[1,0,1],linewidth=0.5)
    plt.plot(np.arange(-2,0,1/20),(cent_TRF[:,1]-cent_TRF_sem[:,1])*sf,color=[1,0,1],linewidth=0.5)
    plt.plot([-2,0],[0,0],'k:')
    ax.axes.get_xaxis().set_ticks([-2,-1,0])
    ax.axes.get_yaxis().set_ticks([])
    plt.ylim([-1.25,1.25])
    plt.title('UV')
    if sf !=1:
        plt.ylabel('Scaled Temporal Filter',fontsize='large', fontweight='bold')
    else:
        plt.ylabel('Mean Temporal Filter',fontsize='large', fontweight='bold')
    # cell type info
    plt.subplot(1,5,3);
    plt.axis('off')
    plt.text(-0.125,0.65,('Cluster '+str(clust)), fontsize=30, fontweight='black')
    plt.text(-0.125,0.4,'N = ' + str(n_cells), fontsize='x-large', fontweight='bold')
    plt.text(-0.125,0.2,'scale factor = ' + str(np.round(sf,decimals=3)), fontsize='large', fontweight='demibold')
    # SRFs
    x = plt.subplot(1,5,4);
    plt.imshow(blue_SRF*sf, origin='lower', cmap='PiYG', clim=[-0.7,0.7]);
    x.axes.get_xaxis().set_ticks([0,40,80])
    x.axes.get_yaxis().set_ticks([0,40,80])
    plt.title('Blue')
    x = plt.subplot(1,5,5);
    plt.imshow(uv_SRF*sf, origin='lower', cmap='PiYG', clim=[-0.7,0.7]);
    x.axes.get_xaxis().set_ticks([0,40,80])
    x.axes.get_yaxis().set_ticks([])
    plt.title('UV')
    if sf !=1:
        plt.ylabel('Scaled Spatial RF',fontsize='large', fontweight='bold')
    else:
        plt.ylabel('Mean Spatial RF',fontsize='large', fontweight='bold')

    # save as PDF and close drawn figure
    # fname = '/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/cluster summaries/cluster_'+str(clust)+'.pdf'
    # sumfig.savefig(fname, format='pdf', orientation='landscape')
    # plt.close()
#%% STUFF AFTER HERE IS NOT IN ACTIVE USE
#%% Failed clustering methods and code to calculate Euclidean distance

# Cluster flattened culled responses using HDBSCAN method
# from sklearn.cluster import HDBSCAN
# hdb = HDBSCAN(min_cluster_size=3)
# clust_assignments = hdb.fit_predict(culled_small_flat)

# Cluster with DBSCAN method
# from sklearn.cluster import DBSCAN
# db = DBSCAN(eps=105, min_samples=2)
# clust_assignments = db.fit_predict(culled_small_flat)

# Cluster with MeanShift method
# from sklearn.cluster import MeanShift, estimate_bandwidth
# ms = MeanShift(bandwidth=estimate_bandwidth(culled_small_flat,quantile=0.5))
# clust_assignments = ms.fit_predict(culled_small_flat)

np.linalg.norm(culled_small_flat[34,:]-culled_small_flat[105,:])

#%%
label_roi_ids = scraped_log[scraped_log[:,14] == label,1]
label_fly_ids = np.zeros(len(label_roi_ids),dtype='int')
for n in range(0,len(label_roi_ids)):
     label_fly_ids[n] = label_roi_ids[n][3:6]
unique_flies = np.unique(label_fly_ids)

for fly in range(0,len(unique_flies)):
    fly_cell_coords = label_pc_coords[label_fly_ids == unique_flies[fly],:]
    if len(fly_cell_coords) > 1:
        fly_dist = []
        for main_cell in range(0,fly_cell_coords.shape[0]):
            for second_cell in range(main_cell+1,fly_cell_coords.shape[0]):
                fly_dist = np.append(fly_dist, np.linalg.norm(fly_cell_coords[main_cell,:]-fly_cell_coords[second_cell,:]))


# %%




for cell in range(0,scraped_log.shape[0]):
    if scraped_log[cell,14] == label:
        # this plots all flies on the same axis

            if int(scraped_log[cell,1][3:6]) == unique_flies[fly]:
                plt.scatter(embedding[cell,0],embedding[cell,1],color=[fly/len(unique_flies),0,1-(fly/len(unique_flies))])
                print(scraped_log[cell,1])

#%% Clustering Workshopping

# load pre-culled summary matrices
psd1D = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/culled_spatial_spectra.npy')
psdt = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/culled_temporal_spectra.npy')
before_zc_auc = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/culled_before_last_lobe_AUC.npy')
after_zc_auc = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/culled_last_lobe_AUC.npy')
flick_mod_depths = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/culled_flicker_MDs.npy')
flick_DCs = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/culled_flicker_DCs.npy')
spatial_metrics = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/culled_spatial_metrics.npy')
# WANT TO ADD NONLINEARITY TO FEATURE VECTOR ONCE I HAVE IT?

# build feature vectors for each neuron by sequential subscription; first try with mean of points for which independent measurements are available for blue/UV (can treat these separately later, but overall this would not be expected to add a lot of signal)
# n_pcs = 0
#
# interim_ind = n_pcs+psd1D.shape[0]+psdt.shape[0]+spatial_metrics.shape[0]
# feat_vecs = np.zeros((pc_weights.shape[0], interim_ind+2+4+4))
# # feat_vecs[:,0:n_pcs] = pc_weights[:,0:n_pcs]
# feat_vecs[:,n_pcs:n_pcs+psd1D.shape[0]] = np.transpose(np.mean(psd1D,axis=1))
# feat_vecs[:,n_pcs+psd1D.shape[0]:n_pcs+psd1D.shape[0]+psdt.shape[0]] = np.transpose(np.mean(psdt,axis=1))
# feat_vecs[:,n_pcs+psd1D.shape[0]+psdt.shape[0]:interim_ind] = np.transpose(spatial_metrics)
# feat_vecs[:,interim_ind:interim_ind+1] = np.transpose(np.asarray([np.mean(before_zc_auc,axis=0)]))
# feat_vecs[:,interim_ind+1:interim_ind+2] = np.transpose(np.asarray([np.mean(after_zc_auc,axis=0)]))
# feat_vecs[:,interim_ind+2:interim_ind+6] = np.transpose(flick_mod_depths)
# feat_vecs[:,interim_ind+6:interim_ind+10] = np.transpose(flick_DCs)

# blue/UV separate in feature vector - this code is super shpagett
feat_vecs = np.zeros((pc_weights.shape[0], 2*(psd1D.shape[0]+psdt.shape[0]+2+2)))
feat_vecs[:,:psd1D.shape[0]] = np.transpose(psd1D[:,0,:])/np.amax(psd1D)
feat_vecs[:,psd1D.shape[0]:psd1D.shape[0]+psdt.shape[0]] = np.transpose(psdt[:,0,:])/np.amax(psdt)
feat_vecs[:,psd1D.shape[0]+psdt.shape[0]:psd1D.shape[0]+psdt.shape[0]+1] = 20*np.transpose(np.asarray([before_zc_auc[0,:]]))/np.amax(np.abs(before_zc_auc))
feat_vecs[:,psd1D.shape[0]+psdt.shape[0]+1:psd1D.shape[0]+psdt.shape[0]+2] = 20*np.transpose(np.asarray([after_zc_auc[0,:]]))/np.amax(np.abs(after_zc_auc))
feat_vecs[:,psd1D.shape[0]+psdt.shape[0]+2:psd1D.shape[0]+psdt.shape[0]+3] = np.transpose(np.asarray([psd1D_pow[0,:]]))/np.amax(psd1D_pow)
feat_vecs[:,psd1D.shape[0]+psdt.shape[0]+3:psd1D.shape[0]+psdt.shape[0]+4] = np.transpose(np.asarray([psdt_pow[0,:]]))/np.amax(psd1D_pow)

halfind = psd1D.shape[0]+psdt.shape[0]+2+2
feat_vecs[:,halfind:halfind+psd1D.shape[0]] = np.transpose(psd1D[:,1,:])/np.amax(psd1D)
feat_vecs[:,halfind+psd1D.shape[0]:halfind+psd1D.shape[0]+psdt.shape[0]] = np.transpose(psdt[:,1,:])/np.amax(psdt)
feat_vecs[:,halfind+psd1D.shape[0]+psdt.shape[0]:halfind+psd1D.shape[0]+psdt.shape[0]+1] = 20*np.transpose(np.asarray([before_zc_auc[1,:]]))/np.amax(np.abs(before_zc_auc))
feat_vecs[:,halfind+psd1D.shape[0]+psdt.shape[0]+1:halfind+psd1D.shape[0]+psdt.shape[0]+2] = 20*np.transpose(np.asarray([after_zc_auc[1,:]]))/np.amax(np.abs(after_zc_auc))
feat_vecs[:,halfind+psd1D.shape[0]+psdt.shape[0]+2:halfind+psd1D.shape[0]+psdt.shape[0]+3] = np.transpose(np.asarray([psd1D_pow[1,:]]))/np.amax(psd1D_pow)
feat_vecs[:,halfind+psd1D.shape[0]+psdt.shape[0]+3:halfind+psd1D.shape[0]+psdt.shape[0]+4] = np.transpose(np.asarray([psdt_pow[1,:]]))/np.amax(psd1D_pow)

#%% PCA model on cell type means - this does not look as good because it's essentially reducing the number of samples the PCs are calculated over. PC0 explains more variance here, but the rest of the PCs explain way less (and look worse)

# load a flattened/culled response matrix for label means
flat_label_means = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/culled_flattened_small_label_means.npy')

# PCA on spatially sub-sampled and rotated data (center 40º of STRF)
pca = PCA()
pca.fit_transform(flat_label_means);
slice_ind = 128000 # for reconstructing the input matrix shape

# visualize first n_pcs eigenvectors
n_pcs = 5

strfpc_fig = plt.figure(figsize=(16,20))
tp = np.arange(19,40,4)
for pc in range(0,n_pcs):
    # define the STRF and flicker for the PC
    # pc_STRF = np.reshape(pca.components_[pc][:slice_ind], all_centered_STRFs.shape[0:4])
    pc_STRF = np.reshape(pca.components_[pc][:slice_ind], all_centered_STRFs[20:60,20:60,:,:,:].shape[0:4])
    pc_flick = np.reshape(pca.components_[pc][slice_ind:], all_flick_resp.shape[0:2])
    # plot blue and UV STRF components
    for n in range(0,6):
        plt.subplot(n_pcs+n_pcs,7,((pc)*14)+(n+1));
        x = plt.imshow(pc_STRF[:,:,int(tp[n]),0], origin='lower', cmap='PiYG',clim=[-0.025,0.025]);
        if n != 0:
            x.axes.get_yaxis().set_ticks([])
            x.axes.get_xaxis().set_ticks([])
        else:
            plt.ylabel('PC' + str(pc) + ' Blue')
            x.axes.get_yaxis().set_ticks([0,40])
            x.axes.get_yaxis().set_ticklabels(['20','60'])
            x.axes.get_xaxis().set_ticks([])
        if pc == 0:
            plt.title(str(np.round((-2+((tp[n]+1))/20),2))+' s')
        x = plt.subplot(n_pcs+n_pcs,7,((pc)*14)+(n+8));
        plt.imshow(pc_STRF[:,:,int(tp[n]),1], origin='lower', cmap='PiYG',clim=[-0.025,0.025]);
        if n != 0:
            x.axes.get_yaxis().set_ticks([])
            x.axes.get_xaxis().set_ticks([])
        else:
            plt.ylabel('PC' + str(pc) + ' UV')
            x.axes.get_yaxis().set_ticks([0,40])
            x.axes.get_yaxis().set_ticklabels(['20','60'])
            x.axes.get_xaxis().set_ticks([])
    x = plt.subplot(n_pcs+n_pcs,7,((pc)*14)+7);
    plt.axis('off')
    plt.colorbar(fraction=0.8, aspect=8, ticks=[-0.02,0,0.02], label='PC weight', shrink=1);

flickpc_fig = plt.figure(figsize=(15,3))
# plot flicker components
for pc in range(0,n_pcs):
    # define the STRF and flicker for the PC
    # pc_STRF = np.reshape(pca.components_[pc][:slice_ind], all_centered_STRFs.shape[0:4])
    pc_STRF = np.reshape(pca.components_[pc][:slice_ind], all_centered_STRFs[20:60,20:60,:,:,:].shape[0:4])
    pc_flick = np.reshape(pca.components_[pc][slice_ind:], all_flick_resp.shape[0:2])
    plt.subplot(1,n_pcs,pc+1);
    plt.title('PC' + str(pc))
    for n in range(0,4):
        x = plt.plot(flick_resamp_t,pc_flick[n,:]);
        plt.ylim(-.02,.02)
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

# save as PDF
fname_strf = '/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/label summaries/STRF principal components.pdf'
fname_flick = '/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/label summaries/flicker principal components.pdf'
strfpc_fig.savefig(fname_strf, format='pdf', orientation='portrait')
flickpc_fig.savefig(fname_flick, format='pdf', orientation='landscape')
