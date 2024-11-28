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
from scipy.optimize import curve_fit

def line(x, k, b):
    y = k*x + b
    return (y)

# disable runtime and deprecation warnings - dangerous!
warnings.filterwarnings("ignore")

book = xlrd.open_workbook('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/ME0708_full_log_snap.xls')
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

# load connectome summary metrics - spatial table from Michael
book = xlrd.open_workbook('/Users/tcurrier/Desktop/Clandinin Lab/Connectomics/NernReiser2024 Data/complete_metrics.xls')
sheet = book.sheet_by_name('complete_metrics')
data = [[sheet.cell_value(r, c) for c in range(sheet.ncols)] for r in range(sheet.nrows)]
spatial_data = np.asarray(data)
sd_vars = spatial_data[0,:]

# grab medulla data and remove hemisphere tags from cell type names
spatial_data = spatial_data[spatial_data[:,2]=='ME(R)',:]
for n in range(0,spatial_data.shape[0]):
    spatial_data[n,1] = spatial_data[n,1][:len(spatial_data[n,1])-2]

# manual entry of NT identity for cell types true N>2
type_NTs = np.asarray(['GABA','GABA','GABA','Glu','Glu','ACh','Glu','Glu','Glu','Glu','ACh','ACh','ACh','ACh','Glu','ACh','Glu','GABA','Glu','GABA','GABA','GABA','ACh','ACh','ACh','ACh','ACh','ACh','ACh','ACh','Glu','ACh','ACh','ACh','ACh','ACh','Glu','Glu','ACh','ACh','Glu','ACh','Glu'])

#%% create data vectors with metrics corresponding to each cell type in the data

# pull list of unique cell type labels in my data
utype_indices = np.unique(scraped_log[:,14]) != ''
unique_types = np.unique(scraped_log[:,14])[utype_indices]
Nbytype = np.zeros(len(unique_types))
nbytype = np.zeros(len(unique_types))

# initialize container vectors for each connectome spatial variable
cols_per_cell = np.zeros(len(unique_types))
cells_per_col = np.zeros(len(unique_types))
total_cells = np.zeros(len(unique_types))
total_presyns = np.zeros(len(unique_types))
total_postsyns = np.zeros(len(unique_types))

# loop over cell types and populate vectors
for ind in range(0,len(unique_types)):
    # define cell type to summarize
    label = unique_types[ind]
    # define N on first loop through cell types
    current_cells = scraped_log[scraped_log[:,14] == label,1]
    current_n = len(current_cells)
    Nbytype[ind] = current_n
    nbytype[ind] = len(np.unique(current_cells))
    # manual adjustment for cell types subdivided beyond my ability to discriminate
    if label == 'T4':
        label = 'T4a'
    # grab data for current cell type and add it to metric-specific vector
    cols_per_cell[ind] = spatial_data[np.where(spatial_data[:,1]==label)[0][0], np.where(sd_vars == 'cell_size_cols')[0][0]]
    cells_per_col[ind] = spatial_data[np.where(spatial_data[:,1]==label)[0][0], np.where(sd_vars == 'coverage_factor_trim')[0][0]]
    total_cells[ind] = spatial_data[np.where(spatial_data[:,1]==label)[0][0], np.where(sd_vars == 'population_size')[0][0]]
    total_presyns[ind] = spatial_data[np.where(spatial_data[:,1]==label)[0][0], np.where(sd_vars == 'n_pre')[0][0]]
    total_postsyns[ind] = spatial_data[np.where(spatial_data[:,1]==label)[0][0], np.where(sd_vars == 'n_post')[0][0]]

# load pre-calculated physiology metrics
center_areas = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all_SRF_center_strict_areas.npy')
ellipse_areas = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all_ellipse_model_areas.npy')
llauc = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all_last_lobe_AUC.npy')
llauc_surr = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all_surround_last_lobe_AUC.npy')
llpeak = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all_last_lobe_peak_z.npy')
llpeak_surr = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all_last_lobe_peak_z_SURROUND.npy')
pc_weights = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all_PC_weights.npy')
n_pcs = pc_weights.shape[1]

# pre-compute summary metrics (max over colors) - to preserve sign, take the max of the absolute value, the multiple by the sign of the mean across colors. since there are only two colors, the sign of the mean tells you the sign of the larger response!
llauc = np.nanmax(np.abs(llauc),axis=0) * np.sign(np.mean(llauc,axis=0))
llauc_surr = np.nanmax(np.abs(llauc_surr),axis=0) * np.sign(np.mean(llauc_surr,axis=0))
llpeak = np.nanmax(np.abs(llpeak),axis=0) * np.sign(np.mean(llpeak,axis=0))
llpeak_surr = np.nanmax(np.abs(llpeak_surr),axis=0) * np.sign(np.mean(llpeak_surr,axis=0))

# initialize physiology container vectors
label_area_mean = np.zeros(len(unique_types))
ellipse_area_mean = np.zeros(len(unique_types))
label_llauc_mean = np.zeros(len(unique_types))
label_llauc_surr_mean = np.zeros(len(unique_types))
label_llpeak_mean = np.zeros(len(unique_types))
label_llpeak_surr_mean = np.zeros(len(unique_types))
label_dist_mean = np.zeros(len(unique_types))
label_TRF_dist_mean = np.full(len(unique_types), np.nan, dtype='float')
label_flick_dist_mean = np.full(len(unique_types), np.nan, dtype='float')

for ind in range(0,len(unique_types)):
    # define cell type to summarize
    current_label = unique_types[ind]
    current_cells = scraped_log[scraped_log[:,14] == current_label,1]
    current_n = len(current_cells)
    # RF area
    label_area = center_areas[scraped_log[:,14] == current_label]
    label_area_mean[ind] = np.nanmedian(label_area)
    ellipse_area = ellipse_areas[scraped_log[:,14] == current_label]
    ellipse_area_mean[ind] = np.nanmedian(ellipse_area)
    # TRF AUC and peak
    label_llauc = llauc[scraped_log[:,14] == current_label]
    label_llauc_mean[ind] = np.nanmedian(label_llauc)
    label_llauc_surr = llauc_surr[scraped_log[:,14] == current_label]
    label_llauc_surr_mean[ind] = np.nanmedian(label_llauc_surr)
    label_llpeak = llpeak[scraped_log[:,14] == current_label]
    label_llpeak_mean[ind] = np.nanmedian(label_llpeak)
    label_llpeak_surr = llpeak_surr[scraped_log[:,14] == current_label]
    label_llpeak_surr_mean[ind] = np.nanmedian(label_llpeak_surr)
    # PC space pairwise distance
    label_pc_coords = pc_weights[scraped_log[:,14] == current_label,0:n_pcs]
    all_dist = []
    for main_cell in range(0,label_pc_coords.shape[0]):
        for second_cell in range(main_cell+1,label_pc_coords.shape[0]):
            all_dist = np.append(all_dist, np.linalg.norm(label_pc_coords[main_cell,:]-label_pc_coords[second_cell,:]))
    label_dist_mean[ind] = np.nanmedian(all_dist)
    # dictionary lookups
    try:
        label_TRF_dist_mean[ind] = np.nanmedian(TRF_dict[current_label])
        label_flick_dist_mean[ind] = np.nanmedian(flick_dict[current_label])
    except:
        pass
#%% trim off thresholded cell types in cell types list
# set threshold N and number of shuffles for null distribution
thresh_n = 2
numshuffles = 10000

# trim metadata
thresh_types = unique_types[nbytype>thresh_n]
thresh_Nbytype = Nbytype[nbytype>thresh_n]

# trim off thresholded cell types in connectome vectors
thresh_cols_per_cell = cols_per_cell[nbytype>thresh_n]
thresh_cells_per_col = cells_per_col[nbytype>thresh_n]
thresh_total_cells = total_cells[nbytype>thresh_n]
thresh_total_presyns = total_presyns[nbytype>thresh_n]
thresh_total_postsyns = total_postsyns[nbytype>thresh_n]

# trim off thresholded cell types in physiology vectors
thresh_area_mean = label_area_mean[nbytype>thresh_n]
thresh_ellipse_area_mean = ellipse_area_mean[nbytype>thresh_n]
thresh_llauc_mean = label_llauc_mean[nbytype>thresh_n]
thresh_llauc_surr_mean = label_llauc_surr_mean[nbytype>thresh_n]
thresh_llpeak_mean = label_llpeak_mean[nbytype>thresh_n]
thresh_llpeak_surr_mean = label_llpeak_surr_mean[nbytype>thresh_n]
thresh_label_dist_mean = label_dist_mean[nbytype>thresh_n]
thresh_TRF_dist_mean = label_TRF_dist_mean[nbytype>thresh_n]
thresh_flick_dist_mean = label_flick_dist_mean[nbytype>thresh_n]


#%% optional tresholding of data vectors to only include certain cell types - this is to be run in place of the previous block!
numshuffles = 10000
type_class = 'Dm'

# initialize thresholded variables
thresh_types = []
thresh_Nbytype = []
thresh_cols_per_cell = []
thresh_cells_per_col = []
thresh_total_cells = []
thresh_total_presyns = []
thresh_total_postsyns = []
thresh_area_mean = []
thresh_ellipse_area_mean = []
thresh_llauc_mean = []
thresh_llauc_surr_mean = []
thresh_llpeak_mean = []
thresh_llpeak_surr_mean = []
thresh_label_dist_mean = []

for ind in range(0,len(unique_types)):
    # define cell type to summarize
    current_label = unique_types[ind]
    # check if current type belongs to specified class
    if current_label[:len(type_class)] == type_class:
        # add metadata
        thresh_types = np.append(thresh_types, unique_types[ind])
        thresh_Nbytype = np.append(thresh_Nbytype, Nbytype[ind])
        # add connectome data
        thresh_cols_per_cell = np.append(thresh_cols_per_cell, cols_per_cell[ind])
        thresh_cells_per_col = np.append(thresh_cells_per_col, cells_per_col[ind])
        thresh_total_cells = np.append(thresh_total_cells, total_cells[ind])
        thresh_total_presyns = np.append(thresh_total_presyns, total_presyns[ind])
        thresh_total_postsyns = np.append(thresh_total_postsyns, total_postsyns[ind])
        # add functional data
        thresh_area_mean = np.append(thresh_area_mean, label_area_mean[ind])
        thresh_ellipse_area_mean = np.append(thresh_ellipse_area_mean, ellipse_area_mean[ind])
        thresh_llauc_mean = np.append(thresh_llauc_mean, label_llauc_mean[ind])
        thresh_llauc_surr_mean = np.append(thresh_llauc_surr_mean, label_llauc_surr_mean[ind])
        thresh_llpeak_mean = np.append(thresh_llpeak_mean, label_llpeak_mean[ind])
        thresh_llpeak_surr_mean = np.append(thresh_llpeak_surr_mean, label_llpeak_surr_mean[ind])
        thresh_label_dist_mean = np.append(thresh_label_dist_mean, label_dist_mean[ind])


#%% plotting PC space variance vs. population size - this block will only run if thresh_n is set to 2 or higher!!
distfig = plt.figure(figsize=(15, 18))

# PAIRWISE DISTANCE DISTRIBUTION
var_to_plot = thresh_label_dist_mean
full_var_vec = pc_weights
plt.subplot(3,1,1)
# sort by means
sorted_inds = np.argsort(var_to_plot)
sorted_types = thresh_types[sorted_inds]
for n in range(0,len(sorted_types)):
    current_label = sorted_types[n]
    label_pc_coords = pc_weights[scraped_log[:,14] == current_label,:]
    all_dist = []
    for main_cell in range(0,label_pc_coords.shape[0]):
        for second_cell in range(main_cell+1,label_pc_coords.shape[0]):
            all_dist = np.append(all_dist, np.linalg.norm(label_pc_coords[main_cell,:]-label_pc_coords[second_cell,:]))
    # remove NaNs
    label_v = all_dist[~np.isnan(all_dist)]
    # if len(label_v) >= 6:
    #     plt.boxplot(label_v, positions=[n], widths=0.8, showfliers=False, showcaps=False, whis=[0,100], labels={''})
    # else:
    #     plt.plot([n, n], [np.max(label_v), np.min(label_v)], color='k', linewidth=0.75)
    plt.scatter(np.full(len(label_v),n),label_v,color=[0,0,0],s=8,alpha=0.15)
    plt.text(n, 70, sorted_types[n], rotation='vertical')
    plt.text(n-0.2, 63, str(len(label_v)))
# plot means and plot params
plt.scatter(np.arange(0,len(thresh_types),1), var_to_plot[sorted_inds], color=[0,0,0], s=30)
plt.plot([0,len(sorted_types)],[29,29],'k:')
plt.ylabel('Pairwise Distance in PC Space')

# PAIRWISE DISTANCE VS. NUM CELLS
plt.subplot(3,3,4)
fn_var = thresh_label_dist_mean
cn_var = thresh_total_cells
# remove nans
nl_cn_var = cn_var[~np.isnan(fn_var)]
nl_fn_var = fn_var[~np.isnan(fn_var)]
# correlation coefficient
corrmat = np.corrcoef(np.asarray([nl_cn_var,nl_fn_var]))
fn_cn_R = np.round(corrmat[0,1],2)
# linear fit
xdata = nl_cn_var
ydata = nl_fn_var
p0 = [1, 50] #initial conditions for gradient descent
popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
fn_cn_k = popt[0]
# scatter data
plt.scatter(nl_cn_var,nl_fn_var,color=[0,0,0],s=20)
plt.plot(np.arange(-50,1100,1),np.arange(-50,1100,1)*fn_cn_k+popt[1],'r-')
plt.title('R = ' + str(fn_cn_R) + ', k = ' + str(np.round(fn_cn_k,5)) + ' /cell')
plt.ylabel('Pairwise Distance in PC Space')
plt.xlabel('# Cells')
# shuffle with replacement
shuffs_fn_var = np.random.choice(nl_fn_var, size=(len(nl_fn_var),numshuffles), replace=True)
# find slope for each shuffle
shuffled_slopes = np.zeros(numshuffles)
xdata = nl_cn_var
for n in range(0,numshuffles):
    ydata = shuffs_fn_var[:,n]
    popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
    shuffled_slopes[n] = popt[0]
# calculate correlation coefficient distribution
shuffled_corrs = np.corrcoef((np.append(nl_cn_var.reshape(-1,1), shuffs_fn_var, axis=1)).T)[0,1:]
# plot correlation coefficients
plt.subplot(3,6,11)
plt.title('Corr. Coef. Nulls')
plt.violinplot(shuffled_corrs, positions=[1], showextrema=False, points=200, widths=[0.9])
plt.plot([0.75,1.25],[np.quantile(shuffled_corrs,0.025),np.quantile(shuffled_corrs,0.025)],'k:',linewidth=0.5)
plt.plot([0.75,1.25],[np.quantile(shuffled_corrs,0.975),np.quantile(shuffled_corrs,0.975)],'k:',linewidth=0.5)
plt.scatter(1, fn_cn_R, color=[0.2,0.2,0.2], s=20)
# test for significance in a two-sided test
if np.logical_or(fn_cn_R > np.quantile(shuffled_corrs,0.975), fn_cn_R < np.quantile(shuffled_corrs,0.025)):
    plt.plot(1,0,'k*')
# plot slopes
plt.subplot(3,6,12)
plt.title('Slope Nulls')
plt.violinplot(shuffled_slopes, positions=[1], showextrema=False, points=200, widths=[0.9])
plt.plot([0.75,1.25],[np.quantile(shuffled_slopes,0.025),np.quantile(shuffled_slopes,0.025)],'k:',linewidth=0.5)
plt.plot([0.75,1.25],[np.quantile(shuffled_slopes,0.975),np.quantile(shuffled_slopes,0.975)],'k:',linewidth=0.5)
plt.scatter(1, fn_cn_k, color=[0.2,0.2,0.2], s=20)
# test for significance in a two-sided test
if np.logical_or(fn_cn_k > np.quantile(shuffled_slopes,0.975), fn_cn_k < np.quantile(shuffled_slopes,0.025)):
    plt.plot(1,0,'k*')

# PAIRWISE DISTANCE VS. POSTSYN/COL
plt.subplot(3,3,5)
fn_var = thresh_label_dist_mean
cn_var = (thresh_total_postsyns/thresh_total_cells)*(1/thresh_cols_per_cell)
# remove nans
nl_cn_var = cn_var[~np.isnan(fn_var)]
nl_fn_var = fn_var[~np.isnan(fn_var)]
# correlation coefficient
corrmat = np.corrcoef(np.asarray([nl_cn_var,nl_fn_var]))
fn_cn_R = np.round(corrmat[0,1],2)
# linear fit
xdata = nl_cn_var
ydata = nl_fn_var
p0 = [1, 50] #initial conditions for gradient descent
popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
fn_cn_k = popt[0]
# scatter data
plt.scatter(nl_cn_var,nl_fn_var,color=[0,0,0],s=20)
plt.plot(np.arange(-20,450,1),np.arange(-20,450,1)*fn_cn_k+popt[1],'r-')
plt.title('R = ' + str(fn_cn_R) + ', k = ' + str(np.round(fn_cn_k,5)) + ' /syn/col')
plt.ylabel('Pairwise Distance in PC Space')
plt.xlabel('Postsynapses/Column')
# shuffle with replacement
shuffs_fn_var = np.random.choice(nl_fn_var, size=(len(nl_fn_var),numshuffles), replace=True)
# find slope for each shuffle
shuffled_slopes = np.zeros(numshuffles)
xdata = nl_cn_var
for n in range(0,numshuffles):
    ydata = shuffs_fn_var[:,n]
    popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
    shuffled_slopes[n] = popt[0]
# calculate correlation coefficient distribution
shuffled_corrs = np.corrcoef((np.append(nl_cn_var.reshape(-1,1), shuffs_fn_var, axis=1)).T)[0,1:]
# plot correlation coefficients
plt.subplot(3,6,11)
plt.title('Corr. Coef. Nulls')
plt.violinplot(shuffled_corrs, positions=[2], showextrema=False, points=200, widths=[0.9])
plt.plot([1.75,2.25],[np.quantile(shuffled_corrs,0.025),np.quantile(shuffled_corrs,0.025)],'k:',linewidth=0.5)
plt.plot([1.75,2.25],[np.quantile(shuffled_corrs,0.975),np.quantile(shuffled_corrs,0.975)],'k:',linewidth=0.5)
x = plt.scatter(2, fn_cn_R, color=[0.2,0.2,0.2], s=20)
# set ticks
x.axes.get_xaxis().set_ticks([1,2])
x.axes.get_xaxis().set_ticklabels(['R1','R2']);
# test for significance in a two-sided test
if np.logical_or(fn_cn_R > np.quantile(shuffled_corrs,0.975), fn_cn_R < np.quantile(shuffled_corrs,0.025)):
    plt.plot(2,0,'k*')
# plot slopes
plt.subplot(3,6,12)
plt.title('Slope Nulls')
plt.violinplot(shuffled_slopes, positions=[2], showextrema=False, points=200, widths=[0.9])
plt.plot([1.75,2.25],[np.quantile(shuffled_slopes,0.025),np.quantile(shuffled_slopes,0.025)],'k:',linewidth=0.5)
plt.plot([1.75,2.25],[np.quantile(shuffled_slopes,0.975),np.quantile(shuffled_slopes,0.975)],'k:',linewidth=0.5)
x = plt.scatter(2, fn_cn_k, color=[0.2,0.2,0.2], s=20)
# set ticks
x.axes.get_xaxis().set_ticks([1,2])
x.axes.get_xaxis().set_ticklabels(['k1','k2']);
# test for significance in a two-sided test
if np.logical_or(fn_cn_k > np.quantile(shuffled_slopes,0.975), fn_cn_k < np.quantile(shuffled_slopes,0.025)):
    plt.plot(2,0,'k*')

# PAIRWISE DISTANCE VS. CELLS/COL
plt.subplot(3,3,8)
fn_var = thresh_label_dist_mean
cn_var = thresh_cells_per_col
# remove nans
nl_cn_var = cn_var[~np.isnan(fn_var)]
nl_fn_var = fn_var[~np.isnan(fn_var)]
# correlation coefficient
corrmat = np.corrcoef(np.asarray([nl_cn_var,nl_fn_var]))
fn_cn_R = np.round(corrmat[0,1],2)
# linear fit
xdata = nl_cn_var
ydata = nl_fn_var
p0 = [1, 50] #initial conditions for gradient descent
popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
fn_cn_k = popt[0]
# scatter data
plt.scatter(nl_cn_var,nl_fn_var,color=[0,0,0],s=20)
plt.plot(np.arange(0,7,1),np.arange(0,7,1)*fn_cn_k+popt[1],'r-')
plt.title('R = ' + str(fn_cn_R) + ', k = ' + str(np.round(fn_cn_k,5)) + ' /cell/col')
plt.ylabel('Pairwise Distance in PC Space')
plt.xlabel('Cells/Column')
# shuffle with replacement
shuffs_fn_var = np.random.choice(nl_fn_var, size=(len(nl_fn_var),numshuffles), replace=True)
# find slope for each shuffle
shuffled_slopes = np.zeros(numshuffles)
xdata = nl_cn_var
for n in range(0,numshuffles):
    ydata = shuffs_fn_var[:,n]
    popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
    shuffled_slopes[n] = popt[0]
# calculate correlation coefficient distribution
shuffled_corrs = np.corrcoef((np.append(nl_cn_var.reshape(-1,1), shuffs_fn_var, axis=1)).T)[0,1:]
# plot correlation coefficients
plt.subplot(3,6,17)
plt.title('Corr. Coef. Nulls')
plt.violinplot(shuffled_corrs, positions=[2], showextrema=False, points=200, widths=[0.9])
plt.plot([1.75,2.25],[np.quantile(shuffled_corrs,0.025),np.quantile(shuffled_corrs,0.025)],'k:',linewidth=0.5)
plt.plot([1.75,2.25],[np.quantile(shuffled_corrs,0.975),np.quantile(shuffled_corrs,0.975)],'k:',linewidth=0.5)
x = plt.scatter(2, fn_cn_R, color=[0.2,0.2,0.2], s=20)
# set ticks
x.axes.get_xaxis().set_ticks([2])
x.axes.get_xaxis().set_ticklabels(['R2']);
# test for significance in a two-sided test
if np.logical_or(fn_cn_R > np.quantile(shuffled_corrs,0.975), fn_cn_R < np.quantile(shuffled_corrs,0.025)):
    plt.plot(2,0,'k*')
# plot slopes
plt.subplot(3,6,18)
plt.title('Slope Nulls')
plt.violinplot(shuffled_slopes, positions=[2], showextrema=False, points=200, widths=[0.9])
plt.plot([1.75,2.25],[np.quantile(shuffled_slopes,0.025),np.quantile(shuffled_slopes,0.025)],'k:',linewidth=0.5)
plt.plot([1.75,2.25],[np.quantile(shuffled_slopes,0.975),np.quantile(shuffled_slopes,0.975)],'k:',linewidth=0.5)
x = plt.scatter(2, fn_cn_k, color=[0.2,0.2,0.2], s=20)
# set ticks
x.axes.get_xaxis().set_ticks([2])
x.axes.get_xaxis().set_ticklabels(['k2']);
# test for significance in a two-sided test
if np.logical_or(fn_cn_k > np.quantile(shuffled_slopes,0.975), fn_cn_k < np.quantile(shuffled_slopes,0.025)):
    plt.plot(2,0,'k*')

# save it
fname = '/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all cell summaries/within type variability connectome correlations.pdf'
distfig.savefig(fname, format='pdf', orientation='portrait')

#%% plotting TF peak distributions and relevant correlations
pkaucfig = plt.figure(figsize=(15, 20))

# TF PEAK DISTRIBUTION
var_to_plot = thresh_llpeak_mean
full_var_vec = llpeak
plt.subplot(4,1,1)
# sort by means
sorted_inds = np.argsort(var_to_plot)
sorted_types = thresh_types[sorted_inds]
# plot a box for each cell type meeting threshold
for n in range(0,len(sorted_types)):
    current_label = sorted_types[n]
    current_n = thresh_Nbytype[sorted_inds][n]
    label_v = full_var_vec[scraped_log[:,14] == current_label]
    # remove NaNs
    label_v = label_v[~np.isnan(label_v)]
    # if len(label_v) >= 6:
    #     plt.boxplot(label_v, positions=[n], widths=0.8, showfliers=False, showcaps=False, whis=[0,100], labels={''})
    # else:
    #     plt.plot([n, n], [np.max(label_v), np.min(label_v)], color='k', linewidth=0.75)
    plt.scatter(np.full(len(label_v),n),label_v,color=[0,0,0],s=8,alpha=0.3)
    plt.text(n, 5.5, sorted_types[n], rotation='vertical')
    plt.text(n, 4.5, str(len(label_v)))
# plot means and plot params
plt.scatter(np.arange(0,len(thresh_types),1), var_to_plot[sorted_inds], color=[0,0,0], s=30)
plt.plot([-1,len(var_to_plot)+1],[0,0],'k:')
plt.xlim([-1,len(var_to_plot)+1])
plt.ylabel('Peak Response (z-score)')

# TRF PEAK VS. NUM PRESYN/CELL/COL
plt.subplot(4,3,4)
fn_var = np.abs(thresh_llpeak_mean)
cn_var = ((thresh_total_presyns/thresh_total_cells)*(1/thresh_cols_per_cell))
# remove nans
nl_cn_var = cn_var[~np.isnan(fn_var)]
nl_fn_var = fn_var[~np.isnan(fn_var)]
# correlation coefficient
corrmat = np.corrcoef(np.asarray([nl_cn_var,nl_fn_var]))
fn_cn_R = np.round(corrmat[0,1],2)
# linear fit
xdata = nl_cn_var
ydata = nl_fn_var
p0 = [1, 50] #initial conditions for gradient descent
popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
fn_cn_k = popt[0]
# scatter data
plt.scatter(nl_cn_var,nl_fn_var,color=[0,0,0],s=20)
plt.plot(np.arange(-10,350,1),np.arange(-10,350,1)*fn_cn_k+popt[1],'k-')
plt.title('R = ' + str(fn_cn_R) + ', k = ' + str(np.round(fn_cn_k,5)) + ' z/syn/col')
plt.ylabel('Absolute Peak Response (z-score)')
plt.xlabel('Presynapses/Column')
plt.ylim(0.8,4.5)
# scatter specific cell types in a different color
subs_inds = np.full(len(thresh_types),False)
for t in range(0,len(thresh_types)):
    if cn_var[t] > 50:
        subs_inds[t] = True
plt.scatter(nl_cn_var[subs_inds],nl_fn_var[subs_inds],color=[0.8,0.2,0.6],s=20)
plt.text(cn_var[thresh_types=='C3'],fn_var[thresh_types=='C3']-1,'C3')
plt.text(cn_var[thresh_types=='L1'],fn_var[thresh_types=='L1']-1,'L1')
plt.text(cn_var[thresh_types=='L2'],fn_var[thresh_types=='L2']-1,'L2')
plt.text(cn_var[thresh_types=='L5'],fn_var[thresh_types=='L5']-1,'L5')
plt.text(cn_var[thresh_types=='Mi1'],fn_var[thresh_types=='Mi1']-1,'Mi1')
plt.text(cn_var[thresh_types=='Mi4'],fn_var[thresh_types=='Mi4']-1,'Mi4')
plt.text(cn_var[thresh_types=='Mi9'],fn_var[thresh_types=='Mi9']-1,'Mi9')
plt.text(cn_var[thresh_types=='Tm1'],fn_var[thresh_types=='Tm1']-1,'Tm1')
plt.text(cn_var[thresh_types=='Tm2'],fn_var[thresh_types=='Tm2']-1,'Tm2')
# shuffle with replacement
shuffs_fn_var = np.random.choice(nl_fn_var, size=(len(nl_fn_var),numshuffles), replace=True)
# find slope for each shuffle
shuffled_slopes = np.zeros(numshuffles)
xdata = nl_cn_var
for n in range(0,numshuffles):
    ydata = shuffs_fn_var[:,n]
    popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
    shuffled_slopes[n] = popt[0]
# calculate correlation coefficient distribution
shuffled_corrs = np.corrcoef((np.append(nl_cn_var.reshape(-1,1), shuffs_fn_var, axis=1)).T)[0,1:]
# plot correlation coefficients
plt.subplot(4,6,11)
plt.title('Corr. Coef. Nulls')
plt.violinplot(shuffled_corrs, positions=[1], showextrema=False, points=200, widths=[0.9])
plt.plot([0.75,1.25],[np.quantile(shuffled_corrs,0.025),np.quantile(shuffled_corrs,0.025)],'k:',linewidth=0.5)
plt.plot([0.75,1.25],[np.quantile(shuffled_corrs,0.975),np.quantile(shuffled_corrs,0.975)],'k:',linewidth=0.5)
plt.scatter(1, fn_cn_R, color=[0.2,0.2,0.2], s=20)
# test for significance in a two-sided test
if np.logical_or(fn_cn_R > np.quantile(shuffled_corrs,0.975), fn_cn_R < np.quantile(shuffled_corrs,0.025)):
    plt.plot(1,0,'k*')
# plot slopes
plt.subplot(4,6,12)
plt.title('Slope Nulls')
plt.violinplot(shuffled_slopes, positions=[1], showextrema=False, points=200, widths=[0.9])
plt.plot([0.75,1.25],[np.quantile(shuffled_slopes,0.025),np.quantile(shuffled_slopes,0.025)],'k:',linewidth=0.5)
plt.plot([0.75,1.25],[np.quantile(shuffled_slopes,0.975),np.quantile(shuffled_slopes,0.975)],'k:',linewidth=0.5)
plt.scatter(1, fn_cn_k, color=[0.2,0.2,0.2], s=20)
# test for significance in a two-sided test
if np.logical_or(fn_cn_k > np.quantile(shuffled_slopes,0.975), fn_cn_k < np.quantile(shuffled_slopes,0.025)):
    plt.plot(1,0,'k*')

# TRF PEAK VS. NUM POSTSYN/CELL
plt.subplot(4,3,5)
fn_var = np.abs(thresh_llpeak_mean)
cn_var = (thresh_total_postsyns/thresh_total_cells)
# remove nans
nl_cn_var = cn_var[~np.isnan(fn_var)]
nl_fn_var = fn_var[~np.isnan(fn_var)]
# correlation coefficient
corrmat = np.corrcoef(np.asarray([nl_cn_var,nl_fn_var]))
fn_cn_R = np.round(corrmat[0,1],2)
# linear fit
xdata = nl_cn_var
ydata = nl_fn_var
p0 = [1, 50] #initial conditions for gradient descent
popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
fn_cn_k = popt[0]
# scatter data
plt.scatter(nl_cn_var,nl_fn_var,color=[0,0,0],s=20)
plt.plot(np.arange(-100,4500,1),np.arange(-100,4500,1)*fn_cn_k+popt[1],'k-')
plt.title('R = ' + str(fn_cn_R) + ', k = ' + str(np.round(fn_cn_k,5)) + ' z/syn')
plt.ylabel('Absolute Peak Response (z-score)')
plt.xlabel('Postsynapses/Cell')
plt.ylim(0.8,4.5)
# shuffle with replacement
shuffs_fn_var = np.random.choice(nl_fn_var, size=(len(nl_fn_var),numshuffles), replace=True)
# find slope for each shuffle
shuffled_slopes = np.zeros(numshuffles)
xdata = nl_cn_var
for n in range(0,numshuffles):
    ydata = shuffs_fn_var[:,n]
    popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
    shuffled_slopes[n] = popt[0]
# calculate correlation coefficient distribution
shuffled_corrs = np.corrcoef((np.append(nl_cn_var.reshape(-1,1), shuffs_fn_var, axis=1)).T)[0,1:]
# plot correlation coefficients
plt.subplot(4,6,11)
plt.title('Corr. Coef. Nulls')
plt.violinplot(shuffled_corrs, positions=[2], showextrema=False, points=200, widths=[0.9])
plt.plot([1.75,2.25],[np.quantile(shuffled_corrs,0.025),np.quantile(shuffled_corrs,0.025)],'k:',linewidth=0.5)
plt.plot([1.75,2.25],[np.quantile(shuffled_corrs,0.975),np.quantile(shuffled_corrs,0.975)],'k:',linewidth=0.5)
x = plt.scatter(2, fn_cn_R, color=[0.2,0.2,0.2], s=20)
# set ticks
x.axes.get_xaxis().set_ticks([1,2])
x.axes.get_xaxis().set_ticklabels(['R1','R2']);
# test for significance in a two-sided test
if np.logical_or(fn_cn_R > np.quantile(shuffled_corrs,0.975), fn_cn_R < np.quantile(shuffled_corrs,0.025)):
    plt.plot(2,0,'k*')
# plot slopes
plt.subplot(4,6,12)
plt.title('Slope Nulls')
plt.violinplot(shuffled_slopes, positions=[2], showextrema=False, points=200, widths=[0.9])
plt.plot([1.75,2.25],[np.quantile(shuffled_slopes,0.025),np.quantile(shuffled_slopes,0.025)],'k:',linewidth=0.5)
plt.plot([1.75,2.25],[np.quantile(shuffled_slopes,0.975),np.quantile(shuffled_slopes,0.975)],'k:',linewidth=0.5)
x = plt.scatter(2, fn_cn_k, color=[0.2,0.2,0.2], s=20)
# set ticks
x.axes.get_xaxis().set_ticks([1,2])
x.axes.get_xaxis().set_ticklabels(['k1','k2']);
# test for significance in a two-sided test
if np.logical_or(fn_cn_k > np.quantile(shuffled_slopes,0.975), fn_cn_k < np.quantile(shuffled_slopes,0.025)):
    plt.plot(2,0,'k*')

# TRF PEAK VS. NUM POSTSYN/CELL/COL
plt.subplot(4,3,7)
fn_var = np.abs(thresh_llpeak_mean)
cn_var = (thresh_total_postsyns/thresh_total_cells)*(1/thresh_cols_per_cell)
# remove nans
nl_cn_var = cn_var[~np.isnan(fn_var)]
nl_fn_var = fn_var[~np.isnan(fn_var)]
# correlation coefficient
corrmat = np.corrcoef(np.asarray([nl_cn_var,nl_fn_var]))
fn_cn_R = np.round(corrmat[0,1],2)
# linear fit
xdata = nl_cn_var
ydata = nl_fn_var
p0 = [1, 0] #initial conditions for gradient descent
popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
fn_cn_k = popt[0]
# scatter data
plt.scatter(nl_cn_var,nl_fn_var,color=[0,0,0],s=20)
plt.plot(np.arange(-20,500,1),np.arange(-20,500,1)*fn_cn_k+popt[1],'k-')
plt.title('R = ' + str(fn_cn_R) + ', k = ' + str(np.round(fn_cn_k,5)) + ' z/syn/col')
plt.ylabel('Absolute Peak Response (z-score)')
plt.xlabel('Postynapses/Column')
# shuffle with replacement
shuffs_fn_var = np.random.choice(nl_fn_var, size=(len(nl_fn_var),numshuffles), replace=True)
# find slope for each shuffle
shuffled_slopes = np.zeros(numshuffles)
xdata = nl_cn_var
for n in range(0,numshuffles):
    ydata = shuffs_fn_var[:,n]
    popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
    shuffled_slopes[n] = popt[0]
# calculate correlation coefficient distribution
shuffled_corrs = np.corrcoef((np.append(nl_cn_var.reshape(-1,1), shuffs_fn_var, axis=1)).T)[0,1:]
# plot correlation coefficients
plt.subplot(4,6,17)
plt.title('Corr. Coef. Nulls')
plt.violinplot(shuffled_corrs, positions=[1], showextrema=False, points=200, widths=[0.9])
plt.plot([0.75,1.25],[np.quantile(shuffled_corrs,0.025),np.quantile(shuffled_corrs,0.025)],'k:',linewidth=0.5)
plt.plot([0.75,1.25],[np.quantile(shuffled_corrs,0.975),np.quantile(shuffled_corrs,0.975)],'k:',linewidth=0.5)
plt.scatter(1, fn_cn_R, color=[0.2,0.2,0.2], s=20)
# test for significance in a two-sided test
if np.logical_or(fn_cn_R > np.quantile(shuffled_corrs,0.975), fn_cn_R < np.quantile(shuffled_corrs,0.025)):
    plt.plot(1,0,'k*')
# plot slopes
plt.subplot(4,6,18)
plt.title('Slope Nulls')
plt.violinplot(shuffled_slopes, positions=[1], showextrema=False, points=200, widths=[0.9])
plt.plot([0.75,1.25],[np.quantile(shuffled_slopes,0.025),np.quantile(shuffled_slopes,0.025)],'k:',linewidth=0.5)
plt.plot([0.75,1.25],[np.quantile(shuffled_slopes,0.975),np.quantile(shuffled_slopes,0.975)],'k:',linewidth=0.5)
plt.scatter(1, fn_cn_k, color=[0.2,0.2,0.2], s=20)
# test for significance in a two-sided test
if np.logical_or(fn_cn_k > np.quantile(shuffled_slopes,0.975), fn_cn_k < np.quantile(shuffled_slopes,0.025)):
    plt.plot(1,0,'k*')

# DISTRIBUTION OF CELL SIZES
plt.subplot(4,3,8)
fn_var = thresh_cols_per_cell
cn_var = (thresh_total_presyns/thresh_total_cells)
# remove nans
nl_cn_var = cn_var[~np.isnan(fn_var)]
nl_fn_var = fn_var[~np.isnan(fn_var)]
# correlation coefficient
corrmat = np.corrcoef(np.asarray([nl_cn_var,nl_fn_var]))
fn_cn_R = np.round(corrmat[0,1],2)
# linear fit
xdata = nl_cn_var
ydata = nl_fn_var
p0 = [1, 50] #initial conditions for gradient descent
popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
fn_cn_k = popt[0]
# scatter data
plt.scatter(nl_cn_var,nl_fn_var,color=[0,0,0],s=20)
# scatter specific cell types in a different color
subs_inds = np.full(len(thresh_types),False)
for t in range(0,len(thresh_types)):
    if thresh_cols_per_cell[t] > 20:
        subs_inds[t] = True
plt.scatter(nl_cn_var[subs_inds],nl_fn_var[subs_inds],color=[0.8,0.2,0.6],s=20)
plt.plot(np.arange(-20,800,1),np.arange(-20,800,1)*fn_cn_k+popt[1],'k-')
plt.title('R = ' + str(fn_cn_R) + ', k = ' + str(np.round(fn_cn_k,5)) + ' cols/syn')
plt.ylabel('Arbor Size (columns)')
plt.xlabel('Presynapses/Cell')
plt.text(cn_var[thresh_types=='Pm1'],fn_var[thresh_types=='Pm1']-4,'Pm1')
plt.text(cn_var[thresh_types=='Pm3'],fn_var[thresh_types=='Pm3']-4,'Pm3')
plt.text(cn_var[thresh_types=='Pm5'],fn_var[thresh_types=='Pm5']-4,'Pm5')
plt.text(cn_var[thresh_types=='Dm13'],fn_var[thresh_types=='Dm13']-4,'Dm13')
plt.text(cn_var[thresh_types=='TmY16'],fn_var[thresh_types=='TmY16']-4,'TmY16')
plt.text(cn_var[thresh_types=='Dm16'],fn_var[thresh_types=='Dm16']-4,'Dm16')
# shuffle with replacement
shuffs_fn_var = np.random.choice(nl_fn_var, size=(len(nl_fn_var),numshuffles), replace=True)
# find slope for each shuffle
shuffled_slopes = np.zeros(numshuffles)
xdata = nl_cn_var
for n in range(0,numshuffles):
    ydata = shuffs_fn_var[:,n]
    popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
    shuffled_slopes[n] = popt[0]
# calculate correlation coefficient distribution
shuffled_corrs = np.corrcoef((np.append(nl_cn_var.reshape(-1,1), shuffs_fn_var, axis=1)).T)[0,1:]
# plot correlation coefficients
plt.subplot(4,6,17)
plt.title('Corr. Coef. Nulls')
plt.violinplot(shuffled_corrs, positions=[2], showextrema=False, points=200, widths=[0.9])
plt.plot([1.75,2.25],[np.quantile(shuffled_corrs,0.025),np.quantile(shuffled_corrs,0.025)],'k:',linewidth=0.5)
plt.plot([1.75,2.25],[np.quantile(shuffled_corrs,0.975),np.quantile(shuffled_corrs,0.975)],'k:',linewidth=0.5)
x = plt.scatter(2, fn_cn_R, color=[0.2,0.2,0.2], s=20)
# set ticks
x.axes.get_xaxis().set_ticks([1,2])
x.axes.get_xaxis().set_ticklabels(['R1','R2']);
# test for significance in a two-sided test
if np.logical_or(fn_cn_R > np.quantile(shuffled_corrs,0.975), fn_cn_R < np.quantile(shuffled_corrs,0.025)):
    plt.plot(2,0,'k*')
# plot slopes
plt.subplot(4,6,18)
plt.title('Slope Nulls')
plt.violinplot(shuffled_slopes, positions=[2], showextrema=False, points=200, widths=[0.9])
plt.plot([1.75,2.25],[np.quantile(shuffled_slopes,0.025),np.quantile(shuffled_slopes,0.025)],'k:',linewidth=0.5)
plt.plot([1.75,2.25],[np.quantile(shuffled_slopes,0.975),np.quantile(shuffled_slopes,0.975)],'k:',linewidth=0.5)
x = plt.scatter(2, fn_cn_k, color=[0.2,0.2,0.2], s=20)
# set ticks
x.axes.get_xaxis().set_ticks([1,2])
x.axes.get_xaxis().set_ticklabels(['k1','k2']);
# test for significance in a two-sided test
if np.logical_or(fn_cn_k > np.quantile(shuffled_slopes,0.975), fn_cn_k < np.quantile(shuffled_slopes,0.025)):
    plt.plot(2,0,'k*')

# TRF PEAK VS. NUM PRESYN/CELL
plt.subplot(4,3,10)
fn_var = np.abs(thresh_llpeak_mean)
cn_var = (thresh_total_presyns/thresh_total_cells)
# remove nans
nl_cn_var = cn_var[~np.isnan(fn_var)]
nl_fn_var = fn_var[~np.isnan(fn_var)]
# correlation coefficient
corrmat = np.corrcoef(np.asarray([nl_cn_var,nl_fn_var]))
fn_cn_R = np.round(corrmat[0,1],2)
# linear fit
xdata = nl_cn_var
ydata = nl_fn_var
p0 = [1, 0] #initial conditions for gradient descent
popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
fn_cn_k = popt[0]
# scatter data
plt.scatter(nl_cn_var,nl_fn_var,color=[0,0,0],s=20)
plt.plot(np.arange(-20,800,1),np.arange(-20,800,1)*fn_cn_k+popt[1],'k-')
plt.title('R = ' + str(fn_cn_R) + ', k = ' + str(np.round(fn_cn_k,5)) + ' z/syn')
plt.ylabel('Absolute Peak Response (z-score)')
plt.xlabel('Preynapses/Cell')
plt.ylim(0.8,4.5)
# shuffle with replacement
shuffs_fn_var = np.random.choice(nl_fn_var, size=(len(nl_fn_var),numshuffles), replace=True)
# find slope for each shuffle
shuffled_slopes = np.zeros(numshuffles)
xdata = nl_cn_var
for n in range(0,numshuffles):
    ydata = shuffs_fn_var[:,n]
    popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
    shuffled_slopes[n] = popt[0]
# calculate correlation coefficient distribution
shuffled_corrs = np.corrcoef((np.append(nl_cn_var.reshape(-1,1), shuffs_fn_var, axis=1)).T)[0,1:]
# plot correlation coefficients
plt.subplot(4,6,23)
plt.title('Corr. Coef. Nulls')
plt.violinplot(shuffled_corrs, positions=[1], showextrema=False, points=200, widths=[0.9])
plt.plot([0.75,1.25],[np.quantile(shuffled_corrs,0.025),np.quantile(shuffled_corrs,0.025)],'k:',linewidth=0.5)
plt.plot([0.75,1.25],[np.quantile(shuffled_corrs,0.975),np.quantile(shuffled_corrs,0.975)],'k:',linewidth=0.5)
plt.scatter(1, fn_cn_R, color=[0.2,0.2,0.2], s=20)
# test for significance in a two-sided test
if np.logical_or(fn_cn_R > np.quantile(shuffled_corrs,0.975), fn_cn_R < np.quantile(shuffled_corrs,0.025)):
    plt.plot(1,0,'k*')
# plot slopes
plt.subplot(4,6,24)
plt.title('Slope Nulls')
plt.violinplot(shuffled_slopes, positions=[1], showextrema=False, points=200, widths=[0.9])
plt.plot([0.75,1.25],[np.quantile(shuffled_slopes,0.025),np.quantile(shuffled_slopes,0.025)],'k:',linewidth=0.5)
plt.plot([0.75,1.25],[np.quantile(shuffled_slopes,0.975),np.quantile(shuffled_slopes,0.975)],'k:',linewidth=0.5)
plt.scatter(1, fn_cn_k, color=[0.2,0.2,0.2], s=20)
# test for significance in a two-sided test
if np.logical_or(fn_cn_k > np.quantile(shuffled_slopes,0.975), fn_cn_k < np.quantile(shuffled_slopes,0.025)):
    plt.plot(1,0,'k*')


# TRF PEAK VS. NUM PRESYN/CELL (LARGE CELLS EXCLUDED)
plt.subplot(4,3,11)
fn_var = np.abs(thresh_llpeak_mean)
cn_var = (thresh_total_presyns/thresh_total_cells)
# remove cells with more than 20 cols/cell
nl_cn_var = cn_var[thresh_cols_per_cell<=20]
nl_fn_var = fn_var[thresh_cols_per_cell<=20]
# correlation coefficient
corrmat = np.corrcoef(np.asarray([nl_cn_var,nl_fn_var]))
fn_cn_R = np.round(corrmat[0,1],2)
# linear fit
xdata = nl_cn_var
ydata = nl_fn_var
p0 = [1, 50] #initial conditions for gradient descent
popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
fn_cn_k = popt[0]
# scatter data
plt.scatter(nl_cn_var,nl_fn_var,color=[0,0,0],s=20)
# scatter specific cell types in a different color
subs_inds = np.full(len(thresh_types),False)
for t in range(0,len(thresh_types)):
    if thresh_cols_per_cell[t] > 20:
        subs_inds[t] = True
plt.scatter(cn_var[subs_inds],fn_var[subs_inds],color=[0.8,0.2,0.6],s=20)
plt.plot(np.arange(-20,800,1),np.arange(-20,800,1)*fn_cn_k+popt[1],'k-')
plt.title('R = ' + str(fn_cn_R) + ', k = ' + str(np.round(fn_cn_k,5)) + ' z/syn')
plt.ylabel('Absolute Peak Response (z-score)')
plt.xlabel('Presynapses/Cell')
plt.text(cn_var[thresh_types=='Pm1']-20,fn_var[thresh_types=='Pm1']+0.15,'Pm1')
plt.text(cn_var[thresh_types=='Pm3']-20,fn_var[thresh_types=='Pm3']+0.15,'Pm3')
plt.text(cn_var[thresh_types=='Pm5']-20,fn_var[thresh_types=='Pm5']+0.15,'Pm5')
plt.text(cn_var[thresh_types=='Dm13']-20,fn_var[thresh_types=='Dm13']+0.15,'Dm13')
plt.text(cn_var[thresh_types=='TmY16']-20,fn_var[thresh_types=='TmY16']+0.15,'TmY16')
plt.text(cn_var[thresh_types=='Dm16']-20,fn_var[thresh_types=='Dm16']+0.15,'Dm16')
plt.ylim(0.8,4.5)
# shuffle with replacement
shuffs_fn_var = np.random.choice(nl_fn_var, size=(len(nl_fn_var),numshuffles), replace=True)
# find slope for each shuffle
shuffled_slopes = np.zeros(numshuffles)
xdata = nl_cn_var
for n in range(0,numshuffles):
    ydata = shuffs_fn_var[:,n]
    popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
    shuffled_slopes[n] = popt[0]
# calculate correlation coefficient distribution
shuffled_corrs = np.corrcoef((np.append(nl_cn_var.reshape(-1,1), shuffs_fn_var, axis=1)).T)[0,1:]
# plot correlation coefficients
plt.subplot(4,6,23)
plt.title('Corr. Coef. Nulls')
plt.violinplot(shuffled_corrs, positions=[3], showextrema=False, points=200, widths=[0.9])
plt.plot([2.75,3.25],[np.quantile(shuffled_corrs,0.025),np.quantile(shuffled_corrs,0.025)],'k:',linewidth=0.5)
plt.plot([2.75,3.25],[np.quantile(shuffled_corrs,0.975),np.quantile(shuffled_corrs,0.975)],'k:',linewidth=0.5)
x = plt.scatter(3, fn_cn_R, color=[0.2,0.2,0.2], s=20)
# set ticks
x.axes.get_xaxis().set_ticks([1,2,3])
x.axes.get_xaxis().set_ticklabels(['ER1','IR1','R2']);
# test for significance in a two-sided test
if np.logical_or(fn_cn_R > np.quantile(shuffled_corrs,0.975), fn_cn_R < np.quantile(shuffled_corrs,0.025)):
    plt.plot(3,0,'k*')
# plot slopes
plt.subplot(4,6,24)
plt.title('Slope Nulls')
plt.violinplot(shuffled_slopes, positions=[3], showextrema=False, points=200, widths=[0.9])
plt.plot([2.75,3.25],[np.quantile(shuffled_slopes,0.025),np.quantile(shuffled_slopes,0.025)],'k:',linewidth=0.5)
plt.plot([2.75,3.25],[np.quantile(shuffled_slopes,0.975),np.quantile(shuffled_slopes,0.975)],'k:',linewidth=0.5)
x = plt.scatter(3, fn_cn_k, color=[0.2,0.2,0.2], s=20)
# set ticks
x.axes.get_xaxis().set_ticks([1,2,3])
x.axes.get_xaxis().set_ticklabels(['Ek1','Ik1','k2']);
# test for significance in a two-sided test
if np.logical_or(fn_cn_k > np.quantile(shuffled_slopes,0.975), fn_cn_k < np.quantile(shuffled_slopes,0.025)):
    plt.plot(3,0,'k*')

# save it
fname = '/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all cell summaries/TRF peak connectome correlations.pdf'
pkaucfig.savefig(fname, format='pdf', orientation='portrait')

#%% plotting RF size distribution and relevant correlations
areafig = plt.figure(figsize=(15, 14))

# RF SIZE DISTRIBUTION
var_to_plot = thresh_ellipse_area_mean
full_var_vec = ellipse_areas
plt.subplot(3,1,1)
# sort by means
sorted_inds = np.argsort(var_to_plot)
sorted_types = thresh_types[sorted_inds]
# plot a box for each cell type meeting threshold
for n in range(0,len(sorted_types)):
    current_label = sorted_types[n]
    current_n = thresh_Nbytype[sorted_inds][n]
    label_v = full_var_vec[scraped_log[:,14] == current_label]
    # remove NaNs
    label_v = label_v[~np.isnan(label_v)]
    # if len(label_v) >= 6:
    #     plt.boxplot(label_v, positions=[n], widths=0.8, showfliers=False, showcaps=False, whis=[0,100], labels={''})
    # else:
    #     plt.plot([n, n], [np.max(label_v), np.min(label_v)], color='k', linewidth=0.75)
    plt.scatter(np.full(len(label_v),n),label_v,color=[0,0,0],s=8,alpha=0.3)
    plt.text(n, 920, sorted_types[n], rotation='vertical')
    plt.text(n, 850, str(len(label_v)))
# plot means and plot params
plt.scatter(np.arange(0,len(thresh_types),1), var_to_plot[sorted_inds], color=[0,0,0], s=30)
plt.plot([-1,len(var_to_plot)+1],[0,0],'k:')
plt.xlim([-1,len(var_to_plot)+1])
plt.ylabel('Ellipse Model Area (sq.deg.)')

# RF SIZE VS. NUM COLUMNS PER CELL
plt.subplot(3,3,4)
fn_var = thresh_ellipse_area_mean
cn_var = thresh_cols_per_cell
# remove nans
nl_cn_var = cn_var[~np.isnan(fn_var)]
nl_fn_var = fn_var[~np.isnan(fn_var)]
# correlation coefficient
corrmat = np.corrcoef(np.asarray([nl_cn_var,nl_fn_var]))
fn_cn_R = np.round(corrmat[0,1],2)
# linear fit
xdata = nl_cn_var
ydata = nl_fn_var
p0 = [1, 50] #initial conditions for gradient descent
popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
fn_cn_k = popt[0]
# scatter data
plt.scatter(nl_cn_var,nl_fn_var,color=[0,0,0],s=20)
plt.plot(np.arange(-5,70,1),np.arange(-5,70,1)*fn_cn_k+popt[1],'r-')
plt.title('R = ' + str(fn_cn_R) + ', k = ' + str(np.round(fn_cn_k,2)) + ' sq.deg/col')
plt.ylabel('Ellipse Model Area (sq.deg.)')
plt.xlabel('Arbor Size (columns)')
# scatter specific cell types in a different color
subs_inds = np.full(len(thresh_types),False)
for t in range(0,len(thresh_types)):
    if cn_var[t] > 20:
        subs_inds[t] = True
plt.scatter(cn_var[subs_inds],fn_var[subs_inds],color=[0.8,0.2,0.6],s=20)
plt.text(cn_var[thresh_types=='Pm1']+5,fn_var[thresh_types=='Pm1']+0.15,'Pm1')
plt.text(cn_var[thresh_types=='Pm3']+5,fn_var[thresh_types=='Pm3']+0.15,'Pm3')
plt.text(cn_var[thresh_types=='Pm5']+5,fn_var[thresh_types=='Pm5']+0.15,'Pm5')
plt.text(cn_var[thresh_types=='Dm13']-10,fn_var[thresh_types=='Dm13']-20,'Dm13')
plt.text(cn_var[thresh_types=='TmY16']+5,fn_var[thresh_types=='TmY16']+0.15,'TmY16')
plt.text(cn_var[thresh_types=='Dm16']+5,fn_var[thresh_types=='Dm16']+20,'Dm16')
# shuffle with replacement
shuffs_fn_var = np.random.choice(nl_fn_var, size=(len(nl_fn_var),numshuffles), replace=True)
# find slope for each shuffle
shuffled_slopes = np.zeros(numshuffles)
xdata = nl_cn_var
for n in range(0,numshuffles):
    ydata = shuffs_fn_var[:,n]
    popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
    shuffled_slopes[n] = popt[0]
# calculate correlation coefficient distribution
shuffled_corrs = np.corrcoef((np.append(nl_cn_var.reshape(-1,1), shuffs_fn_var, axis=1)).T)[0,1:]
# plot correlation coefficients
plt.subplot(3,6,11)
plt.title('Corr. Coef. Nulls')
plt.violinplot(shuffled_corrs, positions=[1], showextrema=False, points=200, widths=[0.9])
plt.plot([0.75,1.25],[np.quantile(shuffled_corrs,0.025),np.quantile(shuffled_corrs,0.025)],'k:',linewidth=0.5)
plt.plot([0.75,1.25],[np.quantile(shuffled_corrs,0.975),np.quantile(shuffled_corrs,0.975)],'k:',linewidth=0.5)
plt.scatter(1, fn_cn_R, color=[0.2,0.2,0.2], s=20)
# test for significance in a two-sided test
if np.logical_or(fn_cn_R > np.quantile(shuffled_corrs,0.975), fn_cn_R < np.quantile(shuffled_corrs,0.025)):
    plt.plot(1,0,'k*')
# plot slopes
plt.subplot(3,6,12)
plt.title('Slope Nulls')
plt.violinplot(shuffled_slopes, positions=[1], showextrema=False, points=200, widths=[0.9])
plt.plot([0.75,1.25],[np.quantile(shuffled_slopes,0.025),np.quantile(shuffled_slopes,0.025)],'k:',linewidth=0.5)
plt.plot([0.75,1.25],[np.quantile(shuffled_slopes,0.975),np.quantile(shuffled_slopes,0.975)],'k:',linewidth=0.5)
plt.scatter(1, fn_cn_k, color=[0.2,0.2,0.2], s=20)
# test for significance in a two-sided test
if np.logical_or(fn_cn_k > np.quantile(shuffled_slopes,0.975), fn_cn_k < np.quantile(shuffled_slopes,0.025)):
    plt.plot(1,0,'k*')

# RF SIZE VS. NUM COLUMNS PER CELL (LARGE CELLS EXCLUDED)
plt.subplot(3,3,5)
fn_var = thresh_ellipse_area_mean
cn_var = thresh_cols_per_cell
# threshold on cn_var
fn_var = fn_var[cn_var < 20]
cn_var = cn_var[cn_var < 20]
# remove nans
nl_cn_var = cn_var[~np.isnan(fn_var)]
nl_fn_var = fn_var[~np.isnan(fn_var)]
# correlation coefficient
corrmat = np.corrcoef(np.asarray([nl_cn_var,nl_fn_var]))
fn_cn_R = np.round(corrmat[0,1],2)
# linear fit
xdata = nl_cn_var
ydata = nl_fn_var
p0 = [1, 50] #initial conditions for gradient descent
popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
fn_cn_k = popt[0]
# scatter data
plt.scatter(nl_cn_var,nl_fn_var,color=[0,0,0],s=20)
plt.plot(np.arange(-1,22,1),np.arange(-1,22,1)*fn_cn_k+popt[1],'r-')
plt.title('R = ' + str(fn_cn_R) + ', k = ' + str(np.round(fn_cn_k,2)) + ' sq.deg/cell')
plt.xlabel('Arbor Size (columns)')
# shuffle with replacement
shuffs_fn_var = np.random.choice(nl_fn_var, size=(len(nl_fn_var),numshuffles), replace=True)
# find slope for each shuffle
shuffled_slopes = np.zeros(numshuffles)
xdata = nl_cn_var
for n in range(0,numshuffles):
    ydata = shuffs_fn_var[:,n]
    popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
    shuffled_slopes[n] = popt[0]
# calculate correlation coefficient distribution
shuffled_corrs = np.corrcoef((np.append(nl_cn_var.reshape(-1,1), shuffs_fn_var, axis=1)).T)[0,1:]
# plot correlation coefficients
plt.subplot(3,6,11)
plt.title('Corr. Coef. Nulls')
plt.violinplot(shuffled_corrs, positions=[2], showextrema=False, points=200, widths=[0.9])
plt.plot([1.75,2.25],[np.quantile(shuffled_corrs,0.025),np.quantile(shuffled_corrs,0.025)],'k:',linewidth=0.5)
plt.plot([1.75,2.25],[np.quantile(shuffled_corrs,0.975),np.quantile(shuffled_corrs,0.975)],'k:',linewidth=0.5)
x = plt.scatter(2, fn_cn_R, color=[0.2,0.2,0.2], s=20)
# set ticks
x.axes.get_xaxis().set_ticks([2])
x.axes.get_xaxis().set_ticklabels(['R2']);
# test for significance in a two-sided test
if np.logical_or(fn_cn_R > np.quantile(shuffled_corrs,0.975), fn_cn_R < np.quantile(shuffled_corrs,0.025)):
    plt.plot(2,0,'k*')
# plot slopes
plt.subplot(3,6,12)
plt.title('Slope Nulls')
plt.violinplot(shuffled_slopes, positions=[2], showextrema=False, points=200, widths=[0.9])
plt.plot([1.75,2.25],[np.quantile(shuffled_slopes,0.025),np.quantile(shuffled_slopes,0.025)],'k:',linewidth=0.5)
plt.plot([1.75,2.25],[np.quantile(shuffled_slopes,0.975),np.quantile(shuffled_slopes,0.975)],'k:',linewidth=0.5)
x = plt.scatter(2, fn_cn_k, color=[0.2,0.2,0.2], s=20)
# set ticks
x.axes.get_xaxis().set_ticks([2])
x.axes.get_xaxis().set_ticklabels(['k2']);
# test for significance in a two-sided test
if np.logical_or(fn_cn_k > np.quantile(shuffled_slopes,0.975), fn_cn_k < np.quantile(shuffled_slopes,0.025)):
    plt.plot(2,0,'k*')

# RF SIZE VS. POP SIZE
plt.subplot(3,3,8)
fn_var = thresh_ellipse_area_mean
cn_var = thresh_total_cells
# remove nans
nl_cn_var = cn_var[~np.isnan(fn_var)]
nl_fn_var = fn_var[~np.isnan(fn_var)]
# correlation coefficient
corrmat = np.corrcoef(np.asarray([nl_cn_var,nl_fn_var]))
fn_cn_R = np.round(corrmat[0,1],2)
# linear fit
xdata = nl_cn_var
ydata = nl_fn_var
p0 = [1, 50] #initial conditions for gradient descent
popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
fn_cn_k = popt[0]
# scatter data
plt.scatter(nl_cn_var,nl_fn_var,color=[0,0,0],s=20)
plt.plot(np.arange(-30,1100,1),np.arange(-30,1100,1)*fn_cn_k+popt[1],'r-')
plt.title('R = ' + str(fn_cn_R) + ', k = ' + str(np.round(fn_cn_k,2)) + ' sq.deg/cell')
plt.xlabel('Population Size (cells)')
# shuffle with replacement
shuffs_fn_var = np.random.choice(nl_fn_var, size=(len(nl_fn_var),numshuffles), replace=True)
# find slope for each shuffle
shuffled_slopes = np.zeros(numshuffles)
xdata = nl_cn_var
for n in range(0,numshuffles):
    ydata = shuffs_fn_var[:,n]
    popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
    shuffled_slopes[n] = popt[0]
# calculate correlation coefficient distribution
shuffled_corrs = np.corrcoef((np.append(nl_cn_var.reshape(-1,1), shuffs_fn_var, axis=1)).T)[0,1:]
# plot correlation coefficients
plt.subplot(3,6,17)
plt.title('Corr. Coef. Nulls')
plt.violinplot(shuffled_corrs, positions=[2], showextrema=False, points=200, widths=[0.9])
plt.plot([1.75,2.25],[np.quantile(shuffled_corrs,0.025),np.quantile(shuffled_corrs,0.025)],'k:',linewidth=0.5)
plt.plot([1.75,2.25],[np.quantile(shuffled_corrs,0.975),np.quantile(shuffled_corrs,0.975)],'k:',linewidth=0.5)
x = plt.scatter(2, fn_cn_R, color=[0.2,0.2,0.2], s=20)
# set ticks
x.axes.get_xaxis().set_ticks([2])
x.axes.get_xaxis().set_ticklabels(['R2']);
# test for significance in a two-sided test
if np.logical_or(fn_cn_R > np.quantile(shuffled_corrs,0.975), fn_cn_R < np.quantile(shuffled_corrs,0.025)):
    plt.plot(2,0,'k*')
# plot slopes
plt.subplot(3,6,18)
plt.title('Slope Nulls')
plt.violinplot(shuffled_slopes, positions=[2], showextrema=False, points=200, widths=[0.9])
plt.plot([1.75,2.25],[np.quantile(shuffled_slopes,0.025),np.quantile(shuffled_slopes,0.025)],'k:',linewidth=0.5)
plt.plot([1.75,2.25],[np.quantile(shuffled_slopes,0.975),np.quantile(shuffled_slopes,0.975)],'k:',linewidth=0.5)
x = plt.scatter(2, fn_cn_k, color=[0.2,0.2,0.2], s=20)
# set ticks
x.axes.get_xaxis().set_ticks([2])
x.axes.get_xaxis().set_ticklabels(['k2']);
# test for significance in a two-sided test
if np.logical_or(fn_cn_k > np.quantile(shuffled_slopes,0.975), fn_cn_k < np.quantile(shuffled_slopes,0.025)):
    plt.plot(2,0,'k*')


# SHUFFLE ANALYSIS FOR NULLS
# shuffle without replacement
# shuffs_ellipse_area_mean = np.zeros((len(nl_thresh_ellipse_area_mean),numshuffles))
# rng = np.random.default_rng()
# for n in range(0,numshuffles):
#     rng.shuffle(nl_thresh_ellipse_area_mean)
#     shuffs_ellipse_area_mean[:,n] = nl_thresh_ellipse_area_mean

# save it
fname = '/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all cell summaries/RF area connectome correlations.pdf'
areafig.savefig(fname, format='pdf', orientation='landscape')



#%% UNUSED IN INITIAL SUBMISSION
# open pickle files to load Alex's postsynapse density maps
import pickle
basepath = '/Users/tcurrier/Desktop/Clandinin Lab/Connectomics/NernReiser2024 Data/'
# cell counts
f = open(os.path.join(basepath, 'neurons_number.pickle'),'rb')
cell_counts_dict = pickle.load(f)
f.close()
# postsynapse counts per column as a spatial map - all inputs
f = open(os.path.join(basepath, 'all_layer_syn_count_across_neurons.pickle'),'rb')
EI_maps_dict = pickle.load(f)
f.close()
# load NT-specific maps
f = open(os.path.join(basepath, 'all_NT_syn_map_across_neurons.pickle'),'rb')
nt_maps_dict = pickle.load(f)
f.close()
# define relevant NT dicts
ach_maps_dict = nt_maps_dict['acetylcholine']
glu_maps_dict = nt_maps_dict['glutamate']
gaba_maps_dict = nt_maps_dict['gaba']
hist_maps_dict = nt_maps_dict['histamine']
# generate separate E and I_maps_dict
I_maps_dict = {}
E_maps_dict = {}
for key in EI_maps_dict.keys():
    try:
        # sum glu gaba hist maps to produce overall I map
        glu = glu_maps_dict[key]
        glu[np.isnan(glu)] = 0
        gaba = gaba_maps_dict[key]
        gaba[np.isnan(gaba)] = 0
        hist = hist_maps_dict[key]
        hist[np.isnan(hist)] = 0
        I_maps_dict[key] = glu + gaba + hist
    except:
        glu = glu_maps_dict[key]
        glu[np.isnan(glu)] = 0
        gaba = gaba_maps_dict[key]
        gaba[np.isnan(gaba)] = 0
        hist = np.zeros(EI_maps_dict['Mi1'].shape)
        I_maps_dict[key] = glu + gaba + hist
    # zero NaNs in ach map to make congruent E map
    ach = ach_maps_dict[key]
    ach[np.isnan(ach)] = 0
    E_maps_dict[key] = ach


#%% correlation coefficient between input map and RF sizes for different synapse count thresholds
from skimage.measure import regionprops
import cv2 as cv

# define map data to use and load OSI data
input_maps_dict = E_maps_dict
all_osi = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all_orientation_selectivity_index.npy')

# create a new cell type list for dealing with the T4 subtype issue
map_types = thresh_types
map_types[map_types == 'T4'] = 'T4a'

# container variables for corrcoefs
rel_col_corr = np.zeros(20)
rel_col_corr_thresh = np.zeros(20)
rel_prod_corr = np.zeros(20)
rel_prod_corr_thresh = np.zeros(20)
rel_osi_corr = np.zeros(20)
rel_osi_corr_thresh = np.zeros(20)
abs_col_corr = np.zeros(20)
abs_col_corr_thresh = np.zeros(20)
abs_prod_corr = np.zeros(20)
abs_prod_corr_thresh = np.zeros(20)
abs_osi_corr = np.zeros(20)
abs_osi_corr_thresh = np.zeros(20)

for thr in range(0,20):
    # define thresholds
    pct_max_thresh = 0.05*thr
    count_thresh = 7.5*thr

    # define map size and initialize distribution containers
    mapsize = input_maps_dict['Mi1'].shape[1] #units = columns
    count_maps_dict = {}
    abs_maj_axis = []
    abs_mnr_axis = []
    norm_maps_dict = {}
    rel_maj_axis = []
    rel_mnr_axis = []
    rel_thresh_cols = []
    abs_thresh_cols = []
    maj_axis = []
    mnr_axis = []
    osi_by_type = []

    # loop over dictionary elements and threshold on synapse counts
    for key in input_maps_dict.keys():
        # only process cell types that meet the threshold used for correlation analysis
        if np.any(map_types == key):
            # dims of map: [major axis col, minor axis col]
            current_map = input_maps_dict[key]
            # convert NaNs to 0s and save a copy to a new dict
            current_map[np.isnan(current_map)] = 0
            count_maps_dict[key] = current_map
            # threshold count map and fit ellipse for thresh
            ret, threshim = cv.threshold(current_map,count_thresh,np.nanmax(current_map),0)
            try:
                improps = regionprops(threshim.astype('uint8'))[0]
                # when data falls in a single row, improps returns a 0 for width instead of 1. the max operation inside the append function here replaces 0s with 1s as they arise
                abs_maj_axis = np.append(abs_maj_axis,np.max([improps['major_axis_length'],1]))
                abs_mnr_axis = np.append(abs_mnr_axis,np.max([improps['minor_axis_length'],1]))
            except:
                abs_maj_axis = np.append(abs_maj_axis,np.nan)
                abs_mnr_axis = np.append(abs_mnr_axis,np.nan)
            # normalize full map to peak and save a copy to norm_maps_dict
            norm_maps_dict[key] = current_map/np.nanmax(current_map)
            # threshold normalized map and fit ellipse for thresh
            ret, threshim = cv.threshold(norm_maps_dict[key],pct_max_thresh,1,0)
            improps = regionprops(threshim.astype('uint8'))[0]
            rel_maj_axis = np.append(rel_maj_axis,np.max([improps['major_axis_length'],1]))
            rel_mnr_axis = np.append(rel_mnr_axis,np.max([improps['minor_axis_length'],1]))
            # count number of columns above pct_max_thresh for normalized map and count_thresh for count map
            rel_thresh_cols = np.append(rel_thresh_cols,len(np.where(norm_maps_dict[key] > pct_max_thresh)[0]))
            abs_thresh_cols = np.append(abs_thresh_cols,len(np.where(current_map > count_thresh)[0]))

            # calculate median OSI as in fig 3
            label_os = all_osi[scraped_log[:,14] == key]
            label_areas = ellipse_areas[scraped_log[:,14] == key]
            # NaN-out OS values for cells with area less than area_thresh
            label_os[label_areas < 105] = np.nan
            osi_by_type = np.append(osi_by_type, np.nanmedian(label_os))

    # find corrcoefs
    fn_var = thresh_ellipse_area_mean
    cn_var = rel_thresh_cols
    # remove nans
    nl_fn_var = fn_var[~np.isnan(cn_var)]
    nl_cn_var = cn_var[~np.isnan(cn_var)]
    # correlation coefficient
    corrmat = np.corrcoef(np.asarray([nl_cn_var,nl_fn_var]))
    rel_col_corr[thr] = np.round(corrmat[0,1],2)
    # threshold on cn_var
    nl_fn_var = nl_fn_var[nl_cn_var < 60]
    nl_cn_var = nl_cn_var[nl_cn_var < 60]
    # correlation coefficient
    corrmat = np.corrcoef(np.asarray([nl_cn_var,nl_fn_var]))
    rel_col_corr_thresh[thr] = np.round(corrmat[0,1],2)

    # find corrcoefs
    fn_var = thresh_ellipse_area_mean
    cn_var = rel_maj_axis*rel_mnr_axis
    # remove nans
    nl_fn_var = fn_var[~np.isnan(cn_var)]
    nl_cn_var = cn_var[~np.isnan(cn_var)]
    # correlation coefficient
    corrmat = np.corrcoef(np.asarray([nl_cn_var,nl_fn_var]))
    rel_prod_corr[thr] = np.round(corrmat[0,1],2)
    # threshold on cn_var
    nl_fn_var = nl_fn_var[nl_cn_var < 60]
    nl_cn_var = nl_cn_var[nl_cn_var < 60]
    # correlation coefficient
    corrmat = np.corrcoef(np.asarray([nl_cn_var,nl_fn_var]))
    rel_prod_corr_thresh[thr] = np.round(corrmat[0,1],2)

    # find corrcoefs
    fn_var = osi_by_type
    cn_var = rel_maj_axis/rel_mnr_axis
    # remove nans
    nl_cn_var = cn_var[~np.isnan(fn_var)]
    nl_fn_var = fn_var[~np.isnan(fn_var)]
    # correlation coefficient
    corrmat = np.corrcoef(np.asarray([nl_cn_var,nl_fn_var]))
    rel_osi_corr[thr] = np.round(corrmat[0,1],2)
    # threshold on cn_var
    nl_fn_var = nl_fn_var[nl_cn_var < 6]
    nl_cn_var = nl_cn_var[nl_cn_var < 6]
    # correlation coefficient
    corrmat = np.corrcoef(np.asarray([nl_cn_var,nl_fn_var]))
    rel_osi_corr_thresh[thr] = np.round(corrmat[0,1],2)

    # find corrcoefs
    fn_var = thresh_ellipse_area_mean
    cn_var = abs_thresh_cols
    # remove nans
    nl_fn_var = fn_var[~np.isnan(cn_var)]
    nl_cn_var = cn_var[~np.isnan(cn_var)]
    # correlation coefficient
    corrmat = np.corrcoef(np.asarray([nl_cn_var,nl_fn_var]))
    abs_col_corr[thr] = np.round(corrmat[0,1],2)
    # threshold on cn_var
    nl_fn_var = nl_fn_var[nl_cn_var < 60]
    nl_cn_var = nl_cn_var[nl_cn_var < 60]
    # correlation coefficient
    corrmat = np.corrcoef(np.asarray([nl_cn_var,nl_fn_var]))
    abs_col_corr_thresh[thr] = np.round(corrmat[0,1],2)

    # find corrcoefs
    fn_var = thresh_ellipse_area_mean
    cn_var = abs_maj_axis*abs_mnr_axis
    # remove nans
    nl_fn_var = fn_var[~np.isnan(cn_var)]
    nl_cn_var = cn_var[~np.isnan(cn_var)]
    # correlation coefficient
    corrmat = np.corrcoef(np.asarray([nl_cn_var,nl_fn_var]))
    abs_prod_corr[thr] = np.round(corrmat[0,1],2)
    # threshold on cn_var
    nl_fn_var = nl_fn_var[nl_cn_var < 60]
    nl_cn_var = nl_cn_var[nl_cn_var < 60]
    # correlation coefficient
    corrmat = np.corrcoef(np.asarray([nl_cn_var,nl_fn_var]))
    abs_prod_corr_thresh[thr] = np.round(corrmat[0,1],2)

    # find corrcoefs
    fn_var = osi_by_type
    cn_var = abs_maj_axis/abs_mnr_axis
    # remove nans
    nl_fn_var = fn_var[~np.isnan(cn_var)]
    nl_cn_var = cn_var[~np.isnan(cn_var)]
    nlnl_cn_var = nl_cn_var[~np.isnan(nl_fn_var)]
    nlnl_fn_var = nl_fn_var[~np.isnan(nl_fn_var)]
    # correlation coefficient
    corrmat = np.corrcoef(np.asarray([nlnl_cn_var,nlnl_fn_var]))
    abs_osi_corr[thr] = np.round(corrmat[0,1],2)
    # threshold on cn_var
    nlnl_fn_var = nlnl_fn_var[nlnl_cn_var < 6]
    nlnl_cn_var = nlnl_cn_var[nlnl_cn_var < 6]
    # correlation coefficient
    corrmat = np.corrcoef(np.asarray([nlnl_cn_var,nlnl_fn_var]))
    abs_osi_corr_thresh[thr] = np.round(corrmat[0,1],2)

# plot threshold sequence results
thrseqfig = plt.figure(figsize=(12,8))
plt.subplot(2,3,1)
plt.title('column counting')
plt.xlabel('threshold (''%'' of max)')
plt.ylabel('R')
plt.plot([0,1],[0,0],'k:')
plt.plot(np.arange(0,1,0.05), rel_col_corr[:])
plt.plot(np.arange(0,1,0.05), rel_col_corr_thresh[:])
plt.ylim(-0.25,0.5)
plt.subplot(2,3,2)
plt.title('axis products')
plt.plot([0,1],[0,0],'k:')
plt.plot(np.arange(0,1,0.05), rel_prod_corr[:])
plt.plot(np.arange(0.1,1,0.05), rel_prod_corr_thresh[2:])
plt.ylim(-0.25,0.5)
plt.subplot(2,3,3)
plt.title('axis ratio (vs OSI)')
plt.plot([0,1],[0,0],'k:')
# plt.plot(np.arange(0,1,0.05), rel_osi_corr[:])
plt.plot(np.arange(0,1,0.05), rel_osi_corr_thresh[:])
plt.ylim(-0.25,0.85)

plt.subplot(2,3,4)
plt.xlabel('threshold (raw synapse count)')
plt.ylabel('R')
plt.plot([0,150],[0,0],'k:')
plt.plot(np.arange(0,150,7.5), abs_col_corr[:])
plt.plot(np.arange(0,150,7.5), abs_col_corr_thresh[:])
plt.ylim(-0.25,0.5)
plt.subplot(2,3,5)
plt.plot([0,150],[0,0],'k:')
plt.plot(np.arange(0,150,7.5), abs_prod_corr[:])
plt.plot(np.arange(15,150,7.5), abs_prod_corr_thresh[2:])
plt.ylim(-0.25,0.5)
plt.subplot(2,3,6)
plt.plot([0,150],[0,0],'k:')
# plt.plot(np.arange(0,150,7.5), abs_osi_corr[:])
plt.plot(np.arange(0,150,7.5), abs_osi_corr_thresh[:])
plt.ylim(-0.25,0.85)

# save it
fname = '/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all cell summaries/correlations_by_input_map_threshold.pdf'
thrseqfig.savefig(fname, format='pdf', orientation='portrait')


#%% thresholding Alex's postsynapse count maps and plotting correlations with RF size
from skimage.measure import regionprops
import cv2 as cv

# define map data to use and load OSI data
input_maps_dict = EI_maps_dict
all_osi = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all_orientation_selectivity_index.npy')

# define thresholds
pct_max_thresh = 0.1
count_thresh = 10

# create a new cell type list for dealing with the T4 subtype issue
map_types = thresh_types
map_types[map_types == 'T4'] = 'T4a'

# define map size and initialize distribution containers
mapsize = input_maps_dict['Mi1'].shape[1] #units = columns
count_maps_dict = {}
abs_maj_axis = []
abs_mnr_axis = []
norm_maps_dict = {}
rel_maj_axis = []
rel_mnr_axis = []
rel_thresh_cols = []
abs_thresh_cols = []
maj_axis = []
mnr_axis = []
osi_by_type = []

# loop over dictionary elements and pull the major and minor axis synapse count
for key in input_maps_dict.keys():
    # only process cell types that meet the threshold used for correlation analysis
    if np.any(map_types == key):
        # dims of map: [major axis col, minor axis col]
        current_map = input_maps_dict[key]
        current_N = cell_counts_dict[key]
        # convert NaNs to 0s and save a copy to a new dict
        current_map[np.isnan(current_map)] = 0
        count_maps_dict[key] = current_map
        # threshold count map and fit ellipse for thresh
        ret, threshim = cv.threshold(current_map,count_thresh,np.nanmax(current_map),0)
        try:
            improps = regionprops(threshim.astype('uint8'))[0]
            # when data falls in a single row, improps returns a 0 for width instead of 1. the max operation inside the append function here replaces 0s with 1s as they arise
            abs_maj_axis = np.append(abs_maj_axis,np.max([improps['major_axis_length'],1]))
            abs_mnr_axis = np.append(abs_mnr_axis,np.max([improps['minor_axis_length'],1]))
        except:
            abs_maj_axis = np.append(abs_maj_axis,np.nan)
            abs_mnr_axis = np.append(abs_mnr_axis,np.nan)
        # normalize full map to peak and save a copy to norm_maps_dict
        norm_maps_dict[key] = current_map/np.nanmax(current_map)
        # threshold normalized map and fit ellipse for thresh
        ret, threshim = cv.threshold(norm_maps_dict[key],pct_max_thresh,1,0)
        improps = regionprops(threshim.astype('uint8'))[0]
        rel_maj_axis = np.append(rel_maj_axis,np.max([improps['major_axis_length'],1]))
        rel_mnr_axis = np.append(rel_mnr_axis,np.max([improps['minor_axis_length'],1]))
        # count number of columns above pct_max_thresh for normalized map and count_thresh for count map
        rel_thresh_cols = np.append(rel_thresh_cols,len(np.where(norm_maps_dict[key] > pct_max_thresh)[0]))
        abs_thresh_cols = np.append(abs_thresh_cols,len(np.where(current_map > count_thresh)[0]))
        # save full map axis lengths for potential spatial normalization
        ret, threshim = cv.threshold(norm_maps_dict[key],0.05,1,0)
        improps = regionprops(threshim.astype('uint8'))[0]
        maj_axis = np.append(maj_axis,np.max([improps['major_axis_length'],1]))
        mnr_axis = np.append(mnr_axis,np.max([improps['minor_axis_length'],1]))
        # calculate median OSI as in fig 3
        label_os = all_osi[scraped_log[:,14] == key]
        label_areas = ellipse_areas[scraped_log[:,14] == key]
        # NaN-out OS values for cells with area less than area_thresh
        label_os[label_areas < 105] = np.nan
        osi_by_type = np.append(osi_by_type, np.nanmedian(label_os))

#%% save column count data to use in other analyses (principally, Fig 5F)
np.save('/Users/tcurrier/Desktop/Clandinin Lab/Connectomics/NernReiser2024 Data/better_arbor.npy', rel_thresh_cols)

#%% plot correlations between column count variables or elliptical fits and SRF size
mapcorrfig = plt.figure(figsize=(15, 14))

# RF SIZE VS. THRESH INPUT %
plt.subplot(3,3,1)
fn_var = thresh_ellipse_area_mean
cn_var = rel_thresh_cols
# remove nans
nl_cn_var = cn_var[~np.isnan(fn_var)]
nl_fn_var = fn_var[~np.isnan(fn_var)]
# threshold on cn_var
nl_fn_var = nl_fn_var[nl_cn_var < 60]
nl_cn_var = nl_cn_var[nl_cn_var < 60]
# correlation coefficient
corrmat = np.corrcoef(np.asarray([nl_cn_var,nl_fn_var]))
fn_cn_R = np.round(corrmat[0,1],2)
# linear fit
xdata = nl_cn_var
ydata = nl_fn_var
p0 = [1, 50] #initial conditions for gradient descent
popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
fn_cn_k = popt[0]
# scatter data
plt.scatter(nl_cn_var,nl_fn_var,color=[0,0,0],s=20)
plt.plot(np.arange(-2,62,1),np.arange(-2,62,1)*fn_cn_k+popt[1],'r-')
plt.title('R = ' + str(fn_cn_R) + ', k = ' + str(np.round(fn_cn_k,2)) + ' sq.deg/col')
plt.ylabel('Ellipse Model Area (sq.deg.)')
plt.xlabel('% Thresh. Input Area (columns)')
# shuffle with replacement
shuffs_fn_var = np.random.choice(nl_fn_var, size=(len(nl_fn_var),numshuffles), replace=True)
# find slope for each shuffle
shuffled_slopes = np.zeros(numshuffles)
xdata = nl_cn_var
for n in range(0,numshuffles):
    ydata = shuffs_fn_var[:,n]
    popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
    shuffled_slopes[n] = popt[0]
# calculate correlation coefficient distribution
shuffled_corrs = np.corrcoef((np.append(nl_cn_var.reshape(-1,1), shuffs_fn_var, axis=1)).T)[0,1:]
# plot correlation coefficients
plt.subplot(3,6,5)
plt.title('Corr. Coef. Nulls')
plt.violinplot(shuffled_corrs, positions=[1], showextrema=False, points=200, widths=[0.9])
plt.plot([0.75,1.25],[np.quantile(shuffled_corrs,0.005),np.quantile(shuffled_corrs,0.005)],'k:',linewidth=0.5)
plt.plot([0.75,1.25],[np.quantile(shuffled_corrs,0.995),np.quantile(shuffled_corrs,0.995)],'k:',linewidth=0.5)
plt.scatter(1, fn_cn_R, color=[0.2,0.2,0.2], s=20)
# test for significance in a two-sided test
if np.logical_or(fn_cn_R > np.quantile(shuffled_corrs,0.995), fn_cn_R < np.quantile(shuffled_corrs,0.005)):
    plt.plot(1,0,'k*')
# plot slopes
plt.subplot(3,6,6)
plt.title('Slope Nulls')
plt.violinplot(shuffled_slopes, positions=[1], showextrema=False, points=200, widths=[0.9])
plt.plot([0.75,1.25],[np.quantile(shuffled_slopes,0.005),np.quantile(shuffled_slopes,0.005)],'k:',linewidth=0.5)
plt.plot([0.75,1.25],[np.quantile(shuffled_slopes,0.995),np.quantile(shuffled_slopes,0.995)],'k:',linewidth=0.5)
plt.scatter(1, fn_cn_k, color=[0.2,0.2,0.2], s=20)
# test for significance in a two-sided test
if np.logical_or(fn_cn_k > np.quantile(shuffled_slopes,0.995), fn_cn_k < np.quantile(shuffled_slopes,0.005)):
    plt.plot(1,0,'k*')

# RF SIZE VS. THRESH INPUT COUNTS
plt.subplot(3,3,2)
fn_var = thresh_ellipse_area_mean
cn_var = abs_thresh_cols
# remove nans
nl_cn_var = cn_var[~np.isnan(fn_var)]
nl_fn_var = fn_var[~np.isnan(fn_var)]
# threshold on cn_var
nl_fn_var = nl_fn_var[nl_cn_var < 60]
nl_cn_var = nl_cn_var[nl_cn_var < 60]
# correlation coefficient
corrmat = np.corrcoef(np.asarray([nl_cn_var,nl_fn_var]))
fn_cn_R = np.round(corrmat[0,1],2)
# linear fit
xdata = nl_cn_var
ydata = nl_fn_var
p0 = [1, 50] #initial conditions for gradient descent
popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
fn_cn_k = popt[0]
# scatter data
plt.scatter(nl_cn_var,nl_fn_var,color=[0,0,0],s=20)
plt.plot(np.arange(-2,62,1),np.arange(-2,62,1)*fn_cn_k+popt[1],'r-')
plt.title('R = ' + str(fn_cn_R) + ', k = ' + str(np.round(fn_cn_k,2)) + ' sq.deg/cell')
plt.xlabel('Count Thresh. Input Area (columns)')
# shuffle with replacement
shuffs_fn_var = np.random.choice(nl_fn_var, size=(len(nl_fn_var),numshuffles), replace=True)
# find slope for each shuffle
shuffled_slopes = np.zeros(numshuffles)
xdata = nl_cn_var
for n in range(0,numshuffles):
    ydata = shuffs_fn_var[:,n]
    popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
    shuffled_slopes[n] = popt[0]
# calculate correlation coefficient distribution
shuffled_corrs = np.corrcoef((np.append(nl_cn_var.reshape(-1,1), shuffs_fn_var, axis=1)).T)[0,1:]
# plot correlation coefficients
plt.subplot(3,6,5)
plt.title('Corr. Coef. Nulls')
plt.violinplot(shuffled_corrs, positions=[2], showextrema=False, points=200, widths=[0.9])
plt.plot([1.75,2.25],[np.quantile(shuffled_corrs,0.005),np.quantile(shuffled_corrs,0.005)],'k:',linewidth=0.5)
plt.plot([1.75,2.25],[np.quantile(shuffled_corrs,0.995),np.quantile(shuffled_corrs,0.995)],'k:',linewidth=0.5)
x = plt.scatter(2, fn_cn_R, color=[0.2,0.2,0.2], s=20)
# set ticks
x.axes.get_xaxis().set_ticks([1,2])
x.axes.get_xaxis().set_ticklabels(['R1','R2']);
# test for significance in a two-sided test
if np.logical_or(fn_cn_R > np.quantile(shuffled_corrs,0.995), fn_cn_R < np.quantile(shuffled_corrs,0.005)):
    plt.plot(2,0,'k*')
# plot slopes
plt.subplot(3,6,6)
plt.title('Slope Nulls')
plt.violinplot(shuffled_slopes, positions=[2], showextrema=False, points=200, widths=[0.9])
plt.plot([1.75,2.25],[np.quantile(shuffled_slopes,0.005),np.quantile(shuffled_slopes,0.005)],'k:',linewidth=0.5)
plt.plot([1.75,2.25],[np.quantile(shuffled_slopes,0.995),np.quantile(shuffled_slopes,0.995)],'k:',linewidth=0.5)
x = plt.scatter(2, fn_cn_k, color=[0.2,0.2,0.2], s=20)
# set ticks
x.axes.get_xaxis().set_ticks([1,2])
x.axes.get_xaxis().set_ticklabels(['k1','k2']);
# test for significance in a two-sided test
if np.logical_or(fn_cn_k > np.quantile(shuffled_slopes,0.995), fn_cn_k < np.quantile(shuffled_slopes,0.005)):
    plt.plot(2,0,'k*')



# RF SIZE VS. THRESH INPUT % MAJOR AXIS
plt.subplot(3,3,4)
fn_var = thresh_ellipse_area_mean
cn_var = rel_maj_axis
# remove nans
nl_cn_var = cn_var[~np.isnan(fn_var)]
nl_fn_var = fn_var[~np.isnan(fn_var)]
# threshold on cn_var
nl_fn_var = nl_fn_var[nl_cn_var < 20]
nl_cn_var = nl_cn_var[nl_cn_var < 20]
# correlation coefficient
corrmat = np.corrcoef(np.asarray([nl_cn_var,nl_fn_var]))
fn_cn_R = np.round(corrmat[0,1],2)
# linear fit
xdata = nl_cn_var
ydata = nl_fn_var
p0 = [1, 50] #initial conditions for gradient descent
popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
fn_cn_k = popt[0]
# scatter data
plt.scatter(nl_cn_var,nl_fn_var,color=[0,0,0],s=20)
plt.plot(np.arange(-1,21,1),np.arange(-1,21,1)*fn_cn_k+popt[1],'r-')
plt.title('R = ' + str(fn_cn_R) + ', k = ' + str(np.round(fn_cn_k,2)) + ' sq.deg/col')
plt.ylabel('Ellipse Model Area (sq.deg.)')
plt.xlabel('% Thresh. Major Axis (columns)')
# shuffle with replacement
shuffs_fn_var = np.random.choice(nl_fn_var, size=(len(nl_fn_var),numshuffles), replace=True)
# find slope for each shuffle
shuffled_slopes = np.zeros(numshuffles)
xdata = nl_cn_var
for n in range(0,numshuffles):
    ydata = shuffs_fn_var[:,n]
    popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
    shuffled_slopes[n] = popt[0]
# calculate correlation coefficient distribution
shuffled_corrs = np.corrcoef((np.append(nl_cn_var.reshape(-1,1), shuffs_fn_var, axis=1)).T)[0,1:]
# plot correlation coefficients
plt.subplot(3,6,11)
plt.title('Corr. Coef. Nulls')
plt.violinplot(shuffled_corrs, positions=[1], showextrema=False, points=200, widths=[0.9])
plt.plot([0.75,1.25],[np.quantile(shuffled_corrs,0.005),np.quantile(shuffled_corrs,0.005)],'k:',linewidth=0.5)
plt.plot([0.75,1.25],[np.quantile(shuffled_corrs,0.995),np.quantile(shuffled_corrs,0.995)],'k:',linewidth=0.5)
plt.scatter(1, fn_cn_R, color=[0.2,0.2,0.2], s=20)
# test for significance in a two-sided test
if np.logical_or(fn_cn_R > np.quantile(shuffled_corrs,0.995), fn_cn_R < np.quantile(shuffled_corrs,0.005)):
    plt.plot(1,0,'k*')
# plot slopes
plt.subplot(3,6,12)
plt.title('Slope Nulls')
plt.violinplot(shuffled_slopes, positions=[1], showextrema=False, points=200, widths=[0.9])
plt.plot([0.75,1.25],[np.quantile(shuffled_slopes,0.005),np.quantile(shuffled_slopes,0.005)],'k:',linewidth=0.5)
plt.plot([0.75,1.25],[np.quantile(shuffled_slopes,0.995),np.quantile(shuffled_slopes,0.995)],'k:',linewidth=0.5)
plt.scatter(1, fn_cn_k, color=[0.2,0.2,0.2], s=20)
# test for significance in a two-sided test
if np.logical_or(fn_cn_k > np.quantile(shuffled_slopes,0.995), fn_cn_k < np.quantile(shuffled_slopes,0.005)):
    plt.plot(1,0,'k*')

# RF SIZE VS. THRESH INPUT COUNTS MAJOR AXIS
plt.subplot(3,3,5)
fn_var = thresh_ellipse_area_mean
cn_var = abs_maj_axis
# remove nans
nl_fn_var = fn_var[~np.isnan(cn_var)]
nl_cn_var = cn_var[~np.isnan(cn_var)]
# threshold on cn_var
nl_fn_var = nl_fn_var[nl_cn_var < 20]
nl_cn_var = nl_cn_var[nl_cn_var < 20]
# correlation coefficient
corrmat = np.corrcoef(np.asarray([nl_cn_var,nl_fn_var]))
fn_cn_R = np.round(corrmat[0,1],2)
# linear fit
xdata = nl_cn_var
ydata = nl_fn_var
p0 = [1, 50] #initial conditions for gradient descent
popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
fn_cn_k = popt[0]
# scatter data
plt.scatter(nl_cn_var,nl_fn_var,color=[0,0,0],s=20)
plt.plot(np.arange(-1,21,1),np.arange(-1,21,1)*fn_cn_k+popt[1],'r-')
plt.title('R = ' + str(fn_cn_R) + ', k = ' + str(np.round(fn_cn_k,2)) + ' sq.deg/cell')
plt.xlabel('Count Thresh. Major Axis (columns)')
# shuffle with replacement
shuffs_fn_var = np.random.choice(nl_fn_var, size=(len(nl_fn_var),numshuffles), replace=True)
# find slope for each shuffle
shuffled_slopes = np.zeros(numshuffles)
xdata = nl_cn_var
for n in range(0,numshuffles):
    ydata = shuffs_fn_var[:,n]
    popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
    shuffled_slopes[n] = popt[0]
# calculate correlation coefficient distribution
shuffled_corrs = np.corrcoef((np.append(nl_cn_var.reshape(-1,1), shuffs_fn_var, axis=1)).T)[0,1:]
# plot correlation coefficients
plt.subplot(3,6,11)
plt.title('Corr. Coef. Nulls')
plt.violinplot(shuffled_corrs, positions=[2], showextrema=False, points=200, widths=[0.9])
plt.plot([1.75,2.25],[np.quantile(shuffled_corrs,0.005),np.quantile(shuffled_corrs,0.005)],'k:',linewidth=0.5)
plt.plot([1.75,2.25],[np.quantile(shuffled_corrs,0.995),np.quantile(shuffled_corrs,0.995)],'k:',linewidth=0.5)
x = plt.scatter(2, fn_cn_R, color=[0.2,0.2,0.2], s=20)
# set ticks
x.axes.get_xaxis().set_ticks([1,2])
x.axes.get_xaxis().set_ticklabels(['R1','R2']);
# test for significance in a two-sided test
if np.logical_or(fn_cn_R > np.quantile(shuffled_corrs,0.995), fn_cn_R < np.quantile(shuffled_corrs,0.005)):
    plt.plot(2,0,'k*')
# plot slopes
plt.subplot(3,6,12)
plt.title('Slope Nulls')
plt.violinplot(shuffled_slopes, positions=[2], showextrema=False, points=200, widths=[0.9])
plt.plot([1.75,2.25],[np.quantile(shuffled_slopes,0.005),np.quantile(shuffled_slopes,0.005)],'k:',linewidth=0.5)
plt.plot([1.75,2.25],[np.quantile(shuffled_slopes,0.995),np.quantile(shuffled_slopes,0.995)],'k:',linewidth=0.5)
x = plt.scatter(2, fn_cn_k, color=[0.2,0.2,0.2], s=20)
# set ticks
x.axes.get_xaxis().set_ticks([1,2])
x.axes.get_xaxis().set_ticklabels(['k1','k2']);
# test for significance in a two-sided test
if np.logical_or(fn_cn_k > np.quantile(shuffled_slopes,0.995), fn_cn_k < np.quantile(shuffled_slopes,0.005)):
    plt.plot(2,0,'k*')



# RF SIZE VS. THRESH INPUT % MAJ*MNR AXIS
plt.subplot(3,3,7)
fn_var = thresh_ellipse_area_mean
cn_var = rel_maj_axis*rel_mnr_axis
# remove nans
nl_cn_var = cn_var[~np.isnan(fn_var)]
nl_fn_var = fn_var[~np.isnan(fn_var)]
# threshold on cn_var
nl_fn_var = nl_fn_var[nl_cn_var < 60]
nl_cn_var = nl_cn_var[nl_cn_var < 60]
# correlation coefficient
corrmat = np.corrcoef(np.asarray([nl_cn_var,nl_fn_var]))
fn_cn_R = np.round(corrmat[0,1],2)
# linear fit
xdata = nl_cn_var
ydata = nl_fn_var
p0 = [1, 50] #initial conditions for gradient descent
popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
fn_cn_k = popt[0]
# scatter data
plt.scatter(nl_cn_var,nl_fn_var,color=[0,0,0],s=20)
plt.plot(np.arange(-2,62,1),np.arange(-2,62,1)*fn_cn_k+popt[1],'r-')
plt.title('R = ' + str(fn_cn_R) + ', k = ' + str(np.round(fn_cn_k,2)) + ' sq.deg/col')
plt.ylabel('Ellipse Model Area (sq.deg.)')
plt.xlabel('% Thresh. Major*Minor Axis (columns)')
# shuffle with replacement
shuffs_fn_var = np.random.choice(nl_fn_var, size=(len(nl_fn_var),numshuffles), replace=True)
# find slope for each shuffle
shuffled_slopes = np.zeros(numshuffles)
xdata = nl_cn_var
for n in range(0,numshuffles):
    ydata = shuffs_fn_var[:,n]
    popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
    shuffled_slopes[n] = popt[0]
# calculate correlation coefficient distribution
shuffled_corrs = np.corrcoef((np.append(nl_cn_var.reshape(-1,1), shuffs_fn_var, axis=1)).T)[0,1:]
# plot correlation coefficients
plt.subplot(3,6,17)
plt.title('Corr. Coef. Nulls')
plt.violinplot(shuffled_corrs, positions=[1], showextrema=False, points=200, widths=[0.9])
plt.plot([0.75,1.25],[np.quantile(shuffled_corrs,0.005),np.quantile(shuffled_corrs,0.005)],'k:',linewidth=0.5)
plt.plot([0.75,1.25],[np.quantile(shuffled_corrs,0.995),np.quantile(shuffled_corrs,0.995)],'k:',linewidth=0.5)
plt.scatter(1, fn_cn_R, color=[0.2,0.2,0.2], s=20)
# test for significance in a two-sided test
if np.logical_or(fn_cn_R > np.quantile(shuffled_corrs,0.995), fn_cn_R < np.quantile(shuffled_corrs,0.005)):
    plt.plot(1,0,'k*')
# plot slopes
plt.subplot(3,6,18)
plt.title('Slope Nulls')
plt.violinplot(shuffled_slopes, positions=[1], showextrema=False, points=200, widths=[0.9])
plt.plot([0.75,1.25],[np.quantile(shuffled_slopes,0.005),np.quantile(shuffled_slopes,0.005)],'k:',linewidth=0.5)
plt.plot([0.75,1.25],[np.quantile(shuffled_slopes,0.995),np.quantile(shuffled_slopes,0.995)],'k:',linewidth=0.5)
plt.scatter(1, fn_cn_k, color=[0.2,0.2,0.2], s=20)
# test for significance in a two-sided test
if np.logical_or(fn_cn_k > np.quantile(shuffled_slopes,0.995), fn_cn_k < np.quantile(shuffled_slopes,0.005)):
    plt.plot(1,0,'k*')

# RF SIZE VS. THRESH INPUT COUNTS MAJ*MNR AXIS
plt.subplot(3,3,8)
fn_var = thresh_ellipse_area_mean
cn_var = abs_maj_axis*abs_mnr_axis
# remove nans
nl_fn_var = fn_var[~np.isnan(cn_var)]
nl_cn_var = cn_var[~np.isnan(cn_var)]
# threshold on cn_var
nl_fn_var = nl_fn_var[nl_cn_var < 60]
nl_cn_var = nl_cn_var[nl_cn_var < 60]
# correlation coefficient
corrmat = np.corrcoef(np.asarray([nl_cn_var,nl_fn_var]))
fn_cn_R = np.round(corrmat[0,1],2)
# linear fit
xdata = nl_cn_var
ydata = nl_fn_var
p0 = [1, 50] #initial conditions for gradient descent
popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
fn_cn_k = popt[0]
# scatter data
plt.scatter(nl_cn_var,nl_fn_var,color=[0,0,0],s=20)
plt.plot(np.arange(-2,62,1),np.arange(-2,62,1)*fn_cn_k+popt[1],'r-')
plt.title('R = ' + str(fn_cn_R) + ', k = ' + str(np.round(fn_cn_k,2)) + ' sq.deg/cell')
plt.xlabel('Count Thresh. Major*Minor Axis (columns)')
# shuffle with replacement
shuffs_fn_var = np.random.choice(nl_fn_var, size=(len(nl_fn_var),numshuffles), replace=True)
# find slope for each shuffle
shuffled_slopes = np.zeros(numshuffles)
xdata = nl_cn_var
for n in range(0,numshuffles):
    ydata = shuffs_fn_var[:,n]
    popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
    shuffled_slopes[n] = popt[0]
# calculate correlation coefficient distribution
shuffled_corrs = np.corrcoef((np.append(nl_cn_var.reshape(-1,1), shuffs_fn_var, axis=1)).T)[0,1:]
# plot correlation coefficients
plt.subplot(3,6,17)
plt.title('Corr. Coef. Nulls')
plt.violinplot(shuffled_corrs, positions=[2], showextrema=False, points=200, widths=[0.9])
plt.plot([1.75,2.25],[np.quantile(shuffled_corrs,0.005),np.quantile(shuffled_corrs,0.005)],'k:',linewidth=0.5)
plt.plot([1.75,2.25],[np.quantile(shuffled_corrs,0.995),np.quantile(shuffled_corrs,0.995)],'k:',linewidth=0.5)
x = plt.scatter(2, fn_cn_R, color=[0.2,0.2,0.2], s=20)
# set ticks
x.axes.get_xaxis().set_ticks([1,2])
x.axes.get_xaxis().set_ticklabels(['R1','R2']);
# test for significance in a two-sided test
if np.logical_or(fn_cn_R > np.quantile(shuffled_corrs,0.995), fn_cn_R < np.quantile(shuffled_corrs,0.005)):
    plt.plot(2,0,'k*')
# plot slopes
plt.subplot(3,6,18)
plt.title('Slope Nulls')
plt.violinplot(shuffled_slopes, positions=[2], showextrema=False, points=200, widths=[0.9])
plt.plot([1.75,2.25],[np.quantile(shuffled_slopes,0.005),np.quantile(shuffled_slopes,0.005)],'k:',linewidth=0.5)
plt.plot([1.75,2.25],[np.quantile(shuffled_slopes,0.995),np.quantile(shuffled_slopes,0.995)],'k:',linewidth=0.5)
x = plt.scatter(2, fn_cn_k, color=[0.2,0.2,0.2], s=20)
# set ticks
x.axes.get_xaxis().set_ticks([1,2])
x.axes.get_xaxis().set_ticklabels(['k1','k2']);
# test for significance in a two-sided test
if np.logical_or(fn_cn_k > np.quantile(shuffled_slopes,0.995), fn_cn_k < np.quantile(shuffled_slopes,0.005)):
    plt.plot(2,0,'k*')

# save it
fname = '/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all cell summaries/thresholded_input_maps_correlations.pdf'
mapcorrfig.savefig(fname, format='pdf', orientation='portrait')


#%% plot correlations between elliptical fit axis ratios and median OSI
osicorrfig = plt.figure(figsize=(14, 4))
plt.suptitle('OSI vs. input map axis ratio for peak-relative or absolute count thresholds')

# THRESHOLDED VERSIONS
# OSI VS. THRESH INPUT % - MAJ/MNR AXIS
plt.subplot(1,3,1)
fn_var = osi_by_type
cn_var = rel_maj_axis/rel_mnr_axis
# remove nans
l_fn_var = fn_var[~np.isnan(cn_var)]
l_cn_var = cn_var[~np.isnan(cn_var)]
nl_cn_var = l_cn_var[~np.isnan(l_fn_var)]
nl_fn_var = l_fn_var[~np.isnan(l_fn_var)]
# threshold on cn_var
nl_fn_var = nl_fn_var[nl_cn_var < 6]
nl_cn_var = nl_cn_var[nl_cn_var < 6]
# correlation coefficient
corrmat = np.corrcoef(np.asarray([nl_cn_var,nl_fn_var]))
fn_cn_R = np.round(corrmat[0,1],2)
# linear fit
xdata = nl_cn_var
ydata = nl_fn_var
p0 = [1, 50] #initial conditions for gradient descent
popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
fn_cn_k = popt[0]
# scatter data
plt.scatter(nl_cn_var,nl_fn_var,color=[0,0,0],s=20)
plt.plot(np.arange(0,8,1),np.arange(0,8,1)*fn_cn_k+popt[1],'r-')
plt.title('R = ' + str(fn_cn_R) + ', k = ' + str(np.round(fn_cn_k,2)) + ' OSI/ratio')
plt.ylabel('OSI')
plt.xlabel('Input Map Axis Ratio')
plt.ylim(1,3.25)
plt.text(0.5,1.1,'threshold = ' + str(int(100*pct_max_thresh)) + '%'' of max. input')
# shuffle with replacement
shuffs_fn_var = np.random.choice(nl_fn_var, size=(len(nl_fn_var),numshuffles), replace=True)
# find slope for each shuffle
shuffled_slopes = np.zeros(numshuffles)
xdata = nl_cn_var
for n in range(0,numshuffles):
    ydata = shuffs_fn_var[:,n]
    popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
    shuffled_slopes[n] = popt[0]
# calculate correlation coefficient distribution
shuffled_corrs = np.corrcoef((np.append(nl_cn_var.reshape(-1,1), shuffs_fn_var, axis=1)).T)[0,1:]
# plot correlation coefficients
plt.subplot(1,6,5)
plt.title('Corr. Coef. Nulls')
plt.violinplot(shuffled_corrs, positions=[1], showextrema=False, points=200, widths=[0.9])
plt.plot([0.75,1.25],[np.quantile(shuffled_corrs,0.005),np.quantile(shuffled_corrs,0.005)],'k:',linewidth=0.5)
plt.plot([0.75,1.25],[np.quantile(shuffled_corrs,0.995),np.quantile(shuffled_corrs,0.995)],'k:',linewidth=0.5)
plt.scatter(1, fn_cn_R, color=[0.2,0.2,0.2], s=20)
# test for significance in a two-sided test
if np.logical_or(fn_cn_R > np.quantile(shuffled_corrs,0.995), fn_cn_R < np.quantile(shuffled_corrs,0.005)):
    plt.plot(1,0,'k*')
# plot slopes
plt.subplot(1,6,6)
plt.title('Slope Nulls')
plt.violinplot(shuffled_slopes, positions=[1], showextrema=False, points=200, widths=[0.9])
plt.plot([0.75,1.25],[np.quantile(shuffled_slopes,0.005),np.quantile(shuffled_slopes,0.005)],'k:',linewidth=0.5)
plt.plot([0.75,1.25],[np.quantile(shuffled_slopes,0.995),np.quantile(shuffled_slopes,0.995)],'k:',linewidth=0.5)
plt.scatter(1, fn_cn_k, color=[0.2,0.2,0.2], s=20)
# test for significance in a two-sided test
if np.logical_or(fn_cn_k > np.quantile(shuffled_slopes,0.995), fn_cn_k < np.quantile(shuffled_slopes,0.005)):
    plt.plot(1,0,'k*')

# OSI VS. THRESH INPUT COUNTS - MAJ/MNR AXIS
plt.subplot(1,3,2)
fn_var = osi_by_type
cn_var = abs_maj_axis/abs_mnr_axis
# remove nans
l_fn_var = fn_var[~np.isnan(cn_var)]
l_cn_var = cn_var[~np.isnan(cn_var)]
nl_cn_var = l_cn_var[~np.isnan(l_fn_var)]
nl_fn_var = l_fn_var[~np.isnan(l_fn_var)]
# threshold on cn_var
nl_fn_var = nl_fn_var[nl_cn_var < 6]
nl_cn_var = nl_cn_var[nl_cn_var < 6]
# correlation coefficient
corrmat = np.corrcoef(np.asarray([nl_cn_var,nl_fn_var]))
fn_cn_R = np.round(corrmat[0,1],2)
# linear fit
xdata = nl_cn_var
ydata = nl_fn_var
p0 = [1, 50] #initial conditions for gradient descent
popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
fn_cn_k = popt[0]
# scatter data
plt.scatter(nl_cn_var,nl_fn_var,color=[0,0,0],s=20)
plt.plot(np.arange(0,8,1),np.arange(0,8,1)*fn_cn_k+popt[1],'r-')
plt.title('R = ' + str(fn_cn_R) + ', k = ' + str(np.round(fn_cn_k,2)) + ' OSI/ratio')
plt.xlabel('Input Map Axis Ratio')
plt.ylim(1,3.25)
plt.text(0.5,1.1,'threshold = ' + str(count_thresh) + ' synapses')
# shuffle with replacement
shuffs_fn_var = np.random.choice(nl_fn_var, size=(len(nl_fn_var),numshuffles), replace=True)
# find slope for each shuffle
shuffled_slopes = np.zeros(numshuffles)
xdata = nl_cn_var
for n in range(0,numshuffles):
    ydata = shuffs_fn_var[:,n]
    popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
    shuffled_slopes[n] = popt[0]
# calculate correlation coefficient distribution
shuffled_corrs = np.corrcoef((np.append(nl_cn_var.reshape(-1,1), shuffs_fn_var, axis=1)).T)[0,1:]
# plot correlation coefficients
plt.subplot(1,6,5)
plt.title('Corr. Coef. Nulls')
plt.violinplot(shuffled_corrs, positions=[2], showextrema=False, points=200, widths=[0.9])
plt.plot([1.75,2.25],[np.quantile(shuffled_corrs,0.005),np.quantile(shuffled_corrs,0.005)],'k:',linewidth=0.5)
plt.plot([1.75,2.25],[np.quantile(shuffled_corrs,0.995),np.quantile(shuffled_corrs,0.995)],'k:',linewidth=0.5)
x = plt.scatter(2, fn_cn_R, color=[0.2,0.2,0.2], s=20)
# set ticks
x.axes.get_xaxis().set_ticks([1,2])
x.axes.get_xaxis().set_ticklabels(['R1','R2']);
# test for significance in a two-sided test
if np.logical_or(fn_cn_R > np.quantile(shuffled_corrs,0.995), fn_cn_R < np.quantile(shuffled_corrs,0.005)):
    plt.plot(2,0,'k*')
# plot slopes
plt.subplot(1,6,6)
plt.title('Slope Nulls')
plt.violinplot(shuffled_slopes, positions=[2], showextrema=False, points=200, widths=[0.9])
plt.plot([1.75,2.25],[np.quantile(shuffled_slopes,0.005),np.quantile(shuffled_slopes,0.005)],'k:',linewidth=0.5)
plt.plot([1.75,2.25],[np.quantile(shuffled_slopes,0.995),np.quantile(shuffled_slopes,0.995)],'k:',linewidth=0.5)
x = plt.scatter(2, fn_cn_k, color=[0.2,0.2,0.2], s=20)
# set ticks
x.axes.get_xaxis().set_ticks([1,2])
x.axes.get_xaxis().set_ticklabels(['k1','k2']);
# test for significance in a two-sided test
if np.logical_or(fn_cn_k > np.quantile(shuffled_slopes,0.995), fn_cn_k < np.quantile(shuffled_slopes,0.005)):
    plt.plot(2,0,'k*')

# save it
fname = '/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all cell summaries/osi_vs_thresholded_input_map_axis_ratio.pdf'
osicorrfig.savefig(fname, format='pdf', orientation='landscape')


#%% plot maps and threshold fit contours for example cell types in Fig 5

# define linspace for fitted contour plotting
t = np.linspace(0, 2*np.pi, 1000)

# normalized maps
exmapfig = plt.figure(figsize=(22, 10))
plt.suptitle('Example neuron synaptic input maps, peak-normalized and absolute')
plt.subplot(2,5,1)
plt.imshow(norm_maps_dict['L1'][6:34,6:34], cmap='binary')
plt.plot(13+(rel_mnr_axis[map_types == 'L1']/2)*np.sin(t), 13+(rel_maj_axis[map_types == 'L1']/2)*np.cos(t), 'r-', linewidth=0.5)
plt.subplot(2,5,2)
plt.imshow(norm_maps_dict['Mi2'][6:34,6:34], cmap='binary')
plt.plot(13+(rel_mnr_axis[map_types == 'Mi2']/2)*np.sin(t), 13+(rel_maj_axis[map_types == 'Mi2']/2)*np.cos(t), 'r-', linewidth=0.5)
plt.subplot(2,5,3)
plt.imshow(norm_maps_dict['Pm5'][6:34,6:34], cmap='binary')
plt.plot(13+(rel_mnr_axis[map_types == 'Pm5']/2)*np.sin(t), 13+(rel_maj_axis[map_types == 'Pm5']/2)*np.cos(t), 'r-', linewidth=0.5)
plt.subplot(2,5,4)
plt.imshow(norm_maps_dict['Dm13'][6:34,6:34], cmap='binary')
plt.plot(13+(rel_mnr_axis[map_types == 'Dm13']/2)*np.sin(t), 13+(rel_maj_axis[map_types == 'Dm13']/2)*np.cos(t), 'r-', linewidth=0.5)
plt.subplot(2,5,5)
plt.axis('off')
plt.colorbar(fraction=1.1, aspect=8, label='Postsynapse Density', shrink=0.7)

# absolute count maps
plt.subplot(2,5,6)
plt.imshow(count_maps_dict['L1'][6:34,6:34], cmap='binary', clim=(0,100))
plt.plot(13+(abs_mnr_axis[map_types == 'L1']/2)*np.sin(t), 13+(abs_maj_axis[map_types == 'L1']/2)*np.cos(t), 'b-', linewidth=0.5)
plt.ylabel('Major Axis (columns)')
plt.xlabel('Minor Axis (columns)')
plt.title('L1')
plt.subplot(2,5,7)
plt.imshow(count_maps_dict['Mi2'][6:34,6:34], cmap='binary', clim=(0,100))
plt.plot(13+(abs_mnr_axis[map_types == 'Mi2']/2)*np.sin(t), 13+(abs_maj_axis[map_types == 'Mi2']/2)*np.cos(t), 'b-', linewidth=0.5)
plt.title('Mi2')
plt.subplot(2,5,8)
plt.imshow(count_maps_dict['Pm5'][6:34,6:34], cmap='binary', clim=(0,100))
plt.plot(13+(abs_mnr_axis[map_types == 'Pm5']/2)*np.sin(t), 13+(abs_maj_axis[map_types == 'Pm5']/2)*np.cos(t), 'b-', linewidth=0.5)
plt.title('Pm5')
plt.subplot(2,5,9)
plt.imshow(count_maps_dict['Dm13'][6:34,6:34], cmap='binary', clim=(0,100))
plt.plot(13+(abs_mnr_axis[map_types == 'Dm13']/2)*np.sin(t), 13+(abs_maj_axis[map_types == 'Dm13']/2)*np.cos(t), 'b-', linewidth=0.5)
plt.title('Dm13')
plt.subplot(2,5,10)
plt.axis('off')
plt.colorbar(fraction=1.1, aspect=8, label='Postsynapse Count', shrink=0.7)

# save it
fname = '/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/label summaries/example_input_maps.pdf'
exmapfig.savefig(fname, format='pdf', orientation='landscape')


#%% NOT USED
# open pickle files to load the within-type cross-correlation dictionaries
import pickle
basepath = '/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/'
# TRF corrs
f = open(os.path.join(basepath, 'within-type_pairwise_TRF_correlations.pkl'),'rb')
TRF_dict = pickle.load(f)
f.close()
# flicker corrs
f = open(os.path.join(basepath, 'within-type_pairwise_flicker_correlations.pkl'),'rb')
flick_dict = pickle.load(f)
f.close()

#%% NOT USED
# plotting within-type TRF cross-correlation vs. population size - this block will only run if thresh_n is set to 2 or higher!!
trfdistfig = plt.figure(figsize=(15, 18))

# PAIRWISE TRF CORR DISTRIBUTION
var_to_plot = thresh_TRF_dist_mean
full_var_vec = TRF_dict
plt.subplot(3,1,1)
# sort by means
sorted_inds = np.argsort(var_to_plot)
sorted_types = thresh_types[sorted_inds]
for n in range(0,len(sorted_types)):
    current_label = sorted_types[n]
    current_n = thresh_Nbytype[sorted_inds][n]
    label_v = full_var_vec[current_label]
    # remove NaNs
    label_v = label_v[~np.isnan(label_v)]
    if len(label_v) >= 6:
        plt.boxplot(label_v, positions=[n], widths=0.8, showfliers=False, showcaps=False, whis=[0,100], labels={''})
    else:
        plt.plot([n, n], [np.max(label_v), np.min(label_v)], color='k', linewidth=0.75)
    plt.text(n, 1.2, sorted_types[n], rotation='vertical')
    # plt.text(n, 1.2, str(int(current_n)), rotation='vertical')
    plt.text(n-0.2, 1.1, str(len(label_v)))
# plot means and plot params
plt.scatter(np.arange(0,len(thresh_types),1), var_to_plot[sorted_inds], color=[0.2,0.2,0.2], s=20)
plt.ylabel('Within-type TRF cross correlation (R)')

# PAIRWISE DISTANCE VS. NUM CELLS
plt.subplot(3,3,4)
fn_var = thresh_TRF_dist_mean
cn_var = thresh_total_cells
# remove nans
nl_cn_var = cn_var[~np.isnan(fn_var)]
nl_fn_var = fn_var[~np.isnan(fn_var)]
# correlation coefficient
corrmat = np.corrcoef(np.asarray([nl_cn_var,nl_fn_var]))
fn_cn_R = np.round(corrmat[0,1],2)
# linear fit
xdata = nl_cn_var
ydata = nl_fn_var
p0 = [1, 50] #initial conditions for gradient descent
popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
fn_cn_k = popt[0]
# scatter data
plt.scatter(nl_cn_var,nl_fn_var,color=[0,0,0],s=20)
plt.plot(np.arange(-50,1100,1),np.arange(-50,1100,1)*fn_cn_k+popt[1],'r-')
plt.title('R = ' + str(fn_cn_R) + ', k = ' + str(np.round(fn_cn_k,5)) + ' /cell')
plt.ylabel('Within-type TRF R')
plt.xlabel('# Cells')
# shuffle with replacement
shuffs_fn_var = np.random.choice(nl_fn_var, size=(len(nl_fn_var),numshuffles), replace=True)
# find slope for each shuffle
shuffled_slopes = np.zeros(numshuffles)
xdata = nl_cn_var
for n in range(0,numshuffles):
    ydata = shuffs_fn_var[:,n]
    popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
    shuffled_slopes[n] = popt[0]
# calculate correlation coefficient distribution
shuffled_corrs = np.corrcoef((np.append(nl_cn_var.reshape(-1,1), shuffs_fn_var, axis=1)).T)[0,1:]
# plot correlation coefficients
plt.subplot(3,6,11)
plt.title('Corr. Coef. Nulls')
plt.violinplot(shuffled_corrs, positions=[1], showextrema=False, points=200, widths=[0.9])
plt.plot([0.75,1.25],[np.quantile(shuffled_corrs,0.025),np.quantile(shuffled_corrs,0.025)],'k:',linewidth=0.5)
plt.plot([0.75,1.25],[np.quantile(shuffled_corrs,0.975),np.quantile(shuffled_corrs,0.975)],'k:',linewidth=0.5)
plt.scatter(1, fn_cn_R, color=[0.2,0.2,0.2], s=20)
# test for significance in a two-sided test
if np.logical_or(fn_cn_R > np.quantile(shuffled_corrs,0.975), fn_cn_R < np.quantile(shuffled_corrs,0.025)):
    plt.plot(1,0,'k*')
# plot slopes
plt.subplot(3,6,12)
plt.title('Slope Nulls')
plt.violinplot(shuffled_slopes, positions=[1], showextrema=False, points=200, widths=[0.9])
plt.plot([0.75,1.25],[np.quantile(shuffled_slopes,0.025),np.quantile(shuffled_slopes,0.025)],'k:',linewidth=0.5)
plt.plot([0.75,1.25],[np.quantile(shuffled_slopes,0.975),np.quantile(shuffled_slopes,0.975)],'k:',linewidth=0.5)
plt.scatter(1, fn_cn_k, color=[0.2,0.2,0.2], s=20)
# test for significance in a two-sided test
if np.logical_or(fn_cn_k > np.quantile(shuffled_slopes,0.975), fn_cn_k < np.quantile(shuffled_slopes,0.025)):
    plt.plot(1,0,'k*')

# PAIRWISE DISTANCE VS. POSTSYN/COL
plt.subplot(3,3,5)
fn_var = thresh_TRF_dist_mean
cn_var = (thresh_total_postsyns/thresh_total_cells)*(1/thresh_cols_per_cell)
# remove nans
nl_cn_var = cn_var[~np.isnan(fn_var)]
nl_fn_var = fn_var[~np.isnan(fn_var)]
# correlation coefficient
corrmat = np.corrcoef(np.asarray([nl_cn_var,nl_fn_var]))
fn_cn_R = np.round(corrmat[0,1],2)
# linear fit
xdata = nl_cn_var
ydata = nl_fn_var
p0 = [1, 50] #initial conditions for gradient descent
popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
fn_cn_k = popt[0]
# scatter data
plt.scatter(nl_cn_var,nl_fn_var,color=[0,0,0],s=20)
plt.plot(np.arange(-20,450,1),np.arange(-20,450,1)*fn_cn_k+popt[1],'r-')
plt.title('R = ' + str(fn_cn_R) + ', k = ' + str(np.round(fn_cn_k,5)) + ' /syn/col')
plt.ylabel('Within-type TRF R')
plt.xlabel('Postsynapses/Column')
# shuffle with replacement
shuffs_fn_var = np.random.choice(nl_fn_var, size=(len(nl_fn_var),numshuffles), replace=True)
# find slope for each shuffle
shuffled_slopes = np.zeros(numshuffles)
xdata = nl_cn_var
for n in range(0,numshuffles):
    ydata = shuffs_fn_var[:,n]
    popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
    shuffled_slopes[n] = popt[0]
# calculate correlation coefficient distribution
shuffled_corrs = np.corrcoef((np.append(nl_cn_var.reshape(-1,1), shuffs_fn_var, axis=1)).T)[0,1:]
# plot correlation coefficients
plt.subplot(3,6,11)
plt.title('Corr. Coef. Nulls')
plt.violinplot(shuffled_corrs, positions=[2], showextrema=False, points=200, widths=[0.9])
plt.plot([1.75,2.25],[np.quantile(shuffled_corrs,0.025),np.quantile(shuffled_corrs,0.025)],'k:',linewidth=0.5)
plt.plot([1.75,2.25],[np.quantile(shuffled_corrs,0.975),np.quantile(shuffled_corrs,0.975)],'k:',linewidth=0.5)
x = plt.scatter(2, fn_cn_R, color=[0.2,0.2,0.2], s=20)
# set ticks
x.axes.get_xaxis().set_ticks([1,2])
x.axes.get_xaxis().set_ticklabels(['R1','R2']);
# test for significance in a two-sided test
if np.logical_or(fn_cn_R > np.quantile(shuffled_corrs,0.975), fn_cn_R < np.quantile(shuffled_corrs,0.025)):
    plt.plot(2,0,'k*')
# plot slopes
plt.subplot(3,6,12)
plt.title('Slope Nulls')
plt.violinplot(shuffled_slopes, positions=[2], showextrema=False, points=200, widths=[0.9])
plt.plot([1.75,2.25],[np.quantile(shuffled_slopes,0.025),np.quantile(shuffled_slopes,0.025)],'k:',linewidth=0.5)
plt.plot([1.75,2.25],[np.quantile(shuffled_slopes,0.975),np.quantile(shuffled_slopes,0.975)],'k:',linewidth=0.5)
x = plt.scatter(2, fn_cn_k, color=[0.2,0.2,0.2], s=20)
# set ticks
x.axes.get_xaxis().set_ticks([1,2])
x.axes.get_xaxis().set_ticklabels(['k1','k2']);
# test for significance in a two-sided test
if np.logical_or(fn_cn_k > np.quantile(shuffled_slopes,0.975), fn_cn_k < np.quantile(shuffled_slopes,0.025)):
    plt.plot(2,0,'k*')

# PAIRWISE DISTANCE VS. CELLS/COL
plt.subplot(3,3,8)
fn_var = thresh_TRF_dist_mean
cn_var = thresh_cells_per_col
# remove nans
nl_cn_var = cn_var[~np.isnan(fn_var)]
nl_fn_var = fn_var[~np.isnan(fn_var)]
# correlation coefficient
corrmat = np.corrcoef(np.asarray([nl_cn_var,nl_fn_var]))
fn_cn_R = np.round(corrmat[0,1],2)
# linear fit
xdata = nl_cn_var
ydata = nl_fn_var
p0 = [1, 50] #initial conditions for gradient descent
popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
fn_cn_k = popt[0]
# scatter data
plt.scatter(nl_cn_var,nl_fn_var,color=[0,0,0],s=20)
plt.plot(np.arange(0,7,1),np.arange(0,7,1)*fn_cn_k+popt[1],'r-')
plt.title('R = ' + str(fn_cn_R) + ', k = ' + str(np.round(fn_cn_k,5)) + ' /cell/col')
plt.ylabel('Within-type TRF R')
plt.xlabel('Cells/Column')
# shuffle with replacement
shuffs_fn_var = np.random.choice(nl_fn_var, size=(len(nl_fn_var),numshuffles), replace=True)
# find slope for each shuffle
shuffled_slopes = np.zeros(numshuffles)
xdata = nl_cn_var
for n in range(0,numshuffles):
    ydata = shuffs_fn_var[:,n]
    popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
    shuffled_slopes[n] = popt[0]
# calculate correlation coefficient distribution
shuffled_corrs = np.corrcoef((np.append(nl_cn_var.reshape(-1,1), shuffs_fn_var, axis=1)).T)[0,1:]
# plot correlation coefficients
plt.subplot(3,6,17)
plt.title('Corr. Coef. Nulls')
plt.violinplot(shuffled_corrs, positions=[2], showextrema=False, points=200, widths=[0.9])
plt.plot([1.75,2.25],[np.quantile(shuffled_corrs,0.025),np.quantile(shuffled_corrs,0.025)],'k:',linewidth=0.5)
plt.plot([1.75,2.25],[np.quantile(shuffled_corrs,0.975),np.quantile(shuffled_corrs,0.975)],'k:',linewidth=0.5)
x = plt.scatter(2, fn_cn_R, color=[0.2,0.2,0.2], s=20)
# set ticks
x.axes.get_xaxis().set_ticks([2])
x.axes.get_xaxis().set_ticklabels(['R2']);
# test for significance in a two-sided test
if np.logical_or(fn_cn_R > np.quantile(shuffled_corrs,0.975), fn_cn_R < np.quantile(shuffled_corrs,0.025)):
    plt.plot(2,0,'k*')
# plot slopes
plt.subplot(3,6,18)
plt.title('Slope Nulls')
plt.violinplot(shuffled_slopes, positions=[2], showextrema=False, points=200, widths=[0.9])
plt.plot([1.75,2.25],[np.quantile(shuffled_slopes,0.025),np.quantile(shuffled_slopes,0.025)],'k:',linewidth=0.5)
plt.plot([1.75,2.25],[np.quantile(shuffled_slopes,0.975),np.quantile(shuffled_slopes,0.975)],'k:',linewidth=0.5)
x = plt.scatter(2, fn_cn_k, color=[0.2,0.2,0.2], s=20)
# set ticks
x.axes.get_xaxis().set_ticks([2])
x.axes.get_xaxis().set_ticklabels(['k2']);
# test for significance in a two-sided test
if np.logical_or(fn_cn_k > np.quantile(shuffled_slopes,0.975), fn_cn_k < np.quantile(shuffled_slopes,0.025)):
    plt.plot(2,0,'k*')

# save it
fname = '/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all cell summaries/within type TRF similarity connectome correlations.pdf'
trfdistfig.savefig(fname, format='pdf', orientation='portrait')

#%% NOT USED
# plotting within-type flicker cross-correlation vs. population size - this block will only run if thresh_n is set to 2 or higher!!
flickdistfig = plt.figure(figsize=(15, 18))

# PAIRWISE TRF CORR DISTRIBUTION
var_to_plot = thresh_flick_dist_mean
full_var_vec = flick_dict
plt.subplot(3,1,1)
# sort by means
sorted_inds = np.argsort(var_to_plot)
sorted_types = thresh_types[sorted_inds]
for n in range(0,len(sorted_types)):
    current_label = sorted_types[n]
    current_n = thresh_Nbytype[sorted_inds][n]
    label_v = full_var_vec[current_label]
    # remove NaNs
    label_v = label_v[~np.isnan(label_v)]
    if len(label_v) >= 6:
        plt.boxplot(label_v, positions=[n], widths=0.8, showfliers=False, showcaps=False, whis=[0,100], labels={''})
    else:
        plt.plot([n, n], [np.max(label_v), np.min(label_v)], color='k', linewidth=0.75)
    plt.text(n, 1.2, sorted_types[n], rotation='vertical')
    # plt.text(n, 1.2, str(int(current_n)), rotation='vertical')
    plt.text(n-0.2, 1.1, str(len(label_v)))
# plot means and plot params
plt.scatter(np.arange(0,len(thresh_types),1), var_to_plot[sorted_inds], color=[0.2,0.2,0.2], s=20)
plt.ylabel('Within-type flicker cross correlation (R)')

# PAIRWISE DISTANCE VS. NUM CELLS
plt.subplot(3,3,4)
fn_var = thresh_flick_dist_mean
cn_var = thresh_total_cells
# remove nans
nl_cn_var = cn_var[~np.isnan(fn_var)]
nl_fn_var = fn_var[~np.isnan(fn_var)]
# correlation coefficient
corrmat = np.corrcoef(np.asarray([nl_cn_var,nl_fn_var]))
fn_cn_R = np.round(corrmat[0,1],2)
# linear fit
xdata = nl_cn_var
ydata = nl_fn_var
p0 = [1, 50] #initial conditions for gradient descent
popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
fn_cn_k = popt[0]
# scatter data
plt.scatter(nl_cn_var,nl_fn_var,color=[0,0,0],s=20)
plt.plot(np.arange(-50,1100,1),np.arange(-50,1100,1)*fn_cn_k+popt[1],'r-')
plt.title('R = ' + str(fn_cn_R) + ', k = ' + str(np.round(fn_cn_k,5)) + ' /cell')
plt.ylabel('Within-type flicker R')
plt.xlabel('# Cells')
# shuffle with replacement
shuffs_fn_var = np.random.choice(nl_fn_var, size=(len(nl_fn_var),numshuffles), replace=True)
# find slope for each shuffle
shuffled_slopes = np.zeros(numshuffles)
xdata = nl_cn_var
for n in range(0,numshuffles):
    ydata = shuffs_fn_var[:,n]
    popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
    shuffled_slopes[n] = popt[0]
# calculate correlation coefficient distribution
shuffled_corrs = np.corrcoef((np.append(nl_cn_var.reshape(-1,1), shuffs_fn_var, axis=1)).T)[0,1:]
# plot correlation coefficients
plt.subplot(3,6,11)
plt.title('Corr. Coef. Nulls')
plt.violinplot(shuffled_corrs, positions=[1], showextrema=False, points=200, widths=[0.9])
plt.plot([0.75,1.25],[np.quantile(shuffled_corrs,0.025),np.quantile(shuffled_corrs,0.025)],'k:',linewidth=0.5)
plt.plot([0.75,1.25],[np.quantile(shuffled_corrs,0.975),np.quantile(shuffled_corrs,0.975)],'k:',linewidth=0.5)
plt.scatter(1, fn_cn_R, color=[0.2,0.2,0.2], s=20)
# test for significance in a two-sided test
if np.logical_or(fn_cn_R > np.quantile(shuffled_corrs,0.975), fn_cn_R < np.quantile(shuffled_corrs,0.025)):
    plt.plot(1,0,'k*')
# plot slopes
plt.subplot(3,6,12)
plt.title('Slope Nulls')
plt.violinplot(shuffled_slopes, positions=[1], showextrema=False, points=200, widths=[0.9])
plt.plot([0.75,1.25],[np.quantile(shuffled_slopes,0.025),np.quantile(shuffled_slopes,0.025)],'k:',linewidth=0.5)
plt.plot([0.75,1.25],[np.quantile(shuffled_slopes,0.975),np.quantile(shuffled_slopes,0.975)],'k:',linewidth=0.5)
plt.scatter(1, fn_cn_k, color=[0.2,0.2,0.2], s=20)
# test for significance in a two-sided test
if np.logical_or(fn_cn_k > np.quantile(shuffled_slopes,0.975), fn_cn_k < np.quantile(shuffled_slopes,0.025)):
    plt.plot(1,0,'k*')

# PAIRWISE DISTANCE VS. POSTSYN/COL
plt.subplot(3,3,5)
fn_var = thresh_flick_dist_mean
cn_var = (thresh_total_postsyns/thresh_total_cells)*(1/thresh_cols_per_cell)
# remove nans
nl_cn_var = cn_var[~np.isnan(fn_var)]
nl_fn_var = fn_var[~np.isnan(fn_var)]
# correlation coefficient
corrmat = np.corrcoef(np.asarray([nl_cn_var,nl_fn_var]))
fn_cn_R = np.round(corrmat[0,1],2)
# linear fit
xdata = nl_cn_var
ydata = nl_fn_var
p0 = [1, 50] #initial conditions for gradient descent
popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
fn_cn_k = popt[0]
# scatter data
plt.scatter(nl_cn_var,nl_fn_var,color=[0,0,0],s=20)
plt.plot(np.arange(-20,450,1),np.arange(-20,450,1)*fn_cn_k+popt[1],'r-')
plt.title('R = ' + str(fn_cn_R) + ', k = ' + str(np.round(fn_cn_k,5)) + ' /syn/col')
plt.ylabel('Within-type flicker R')
plt.xlabel('Postsynapses/Column')
# shuffle with replacement
shuffs_fn_var = np.random.choice(nl_fn_var, size=(len(nl_fn_var),numshuffles), replace=True)
# find slope for each shuffle
shuffled_slopes = np.zeros(numshuffles)
xdata = nl_cn_var
for n in range(0,numshuffles):
    ydata = shuffs_fn_var[:,n]
    popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
    shuffled_slopes[n] = popt[0]
# calculate correlation coefficient distribution
shuffled_corrs = np.corrcoef((np.append(nl_cn_var.reshape(-1,1), shuffs_fn_var, axis=1)).T)[0,1:]
# plot correlation coefficients
plt.subplot(3,6,11)
plt.title('Corr. Coef. Nulls')
plt.violinplot(shuffled_corrs, positions=[2], showextrema=False, points=200, widths=[0.9])
plt.plot([1.75,2.25],[np.quantile(shuffled_corrs,0.025),np.quantile(shuffled_corrs,0.025)],'k:',linewidth=0.5)
plt.plot([1.75,2.25],[np.quantile(shuffled_corrs,0.975),np.quantile(shuffled_corrs,0.975)],'k:',linewidth=0.5)
x = plt.scatter(2, fn_cn_R, color=[0.2,0.2,0.2], s=20)
# set ticks
x.axes.get_xaxis().set_ticks([1,2])
x.axes.get_xaxis().set_ticklabels(['R1','R2']);
# test for significance in a two-sided test
if np.logical_or(fn_cn_R > np.quantile(shuffled_corrs,0.975), fn_cn_R < np.quantile(shuffled_corrs,0.025)):
    plt.plot(2,0,'k*')
# plot slopes
plt.subplot(3,6,12)
plt.title('Slope Nulls')
plt.violinplot(shuffled_slopes, positions=[2], showextrema=False, points=200, widths=[0.9])
plt.plot([1.75,2.25],[np.quantile(shuffled_slopes,0.025),np.quantile(shuffled_slopes,0.025)],'k:',linewidth=0.5)
plt.plot([1.75,2.25],[np.quantile(shuffled_slopes,0.975),np.quantile(shuffled_slopes,0.975)],'k:',linewidth=0.5)
x = plt.scatter(2, fn_cn_k, color=[0.2,0.2,0.2], s=20)
# set ticks
x.axes.get_xaxis().set_ticks([1,2])
x.axes.get_xaxis().set_ticklabels(['k1','k2']);
# test for significance in a two-sided test
if np.logical_or(fn_cn_k > np.quantile(shuffled_slopes,0.975), fn_cn_k < np.quantile(shuffled_slopes,0.025)):
    plt.plot(2,0,'k*')

# PAIRWISE DISTANCE VS. CELLS/COL
plt.subplot(3,3,8)
fn_var = thresh_flick_dist_mean
cn_var = thresh_cells_per_col
# remove nans
nl_cn_var = cn_var[~np.isnan(fn_var)]
nl_fn_var = fn_var[~np.isnan(fn_var)]
# correlation coefficient
corrmat = np.corrcoef(np.asarray([nl_cn_var,nl_fn_var]))
fn_cn_R = np.round(corrmat[0,1],2)
# linear fit
xdata = nl_cn_var
ydata = nl_fn_var
p0 = [1, 50] #initial conditions for gradient descent
popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
fn_cn_k = popt[0]
# scatter data
plt.scatter(nl_cn_var,nl_fn_var,color=[0,0,0],s=20)
plt.plot(np.arange(0,7,1),np.arange(0,7,1)*fn_cn_k+popt[1],'r-')
plt.title('R = ' + str(fn_cn_R) + ', k = ' + str(np.round(fn_cn_k,5)) + ' /cell/col')
plt.ylabel('Within-type flicker R')
plt.xlabel('Cells/Column')
# shuffle with replacement
shuffs_fn_var = np.random.choice(nl_fn_var, size=(len(nl_fn_var),numshuffles), replace=True)
# find slope for each shuffle
shuffled_slopes = np.zeros(numshuffles)
xdata = nl_cn_var
for n in range(0,numshuffles):
    ydata = shuffs_fn_var[:,n]
    popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
    shuffled_slopes[n] = popt[0]
# calculate correlation coefficient distribution
shuffled_corrs = np.corrcoef((np.append(nl_cn_var.reshape(-1,1), shuffs_fn_var, axis=1)).T)[0,1:]
# plot correlation coefficients
plt.subplot(3,6,17)
plt.title('Corr. Coef. Nulls')
plt.violinplot(shuffled_corrs, positions=[2], showextrema=False, points=200, widths=[0.9])
plt.plot([1.75,2.25],[np.quantile(shuffled_corrs,0.025),np.quantile(shuffled_corrs,0.025)],'k:',linewidth=0.5)
plt.plot([1.75,2.25],[np.quantile(shuffled_corrs,0.975),np.quantile(shuffled_corrs,0.975)],'k:',linewidth=0.5)
x = plt.scatter(2, fn_cn_R, color=[0.2,0.2,0.2], s=20)
# set ticks
x.axes.get_xaxis().set_ticks([2])
x.axes.get_xaxis().set_ticklabels(['R2']);
# test for significance in a two-sided test
if np.logical_or(fn_cn_R > np.quantile(shuffled_corrs,0.975), fn_cn_R < np.quantile(shuffled_corrs,0.025)):
    plt.plot(2,0,'k*')
# plot slopes
plt.subplot(3,6,18)
plt.title('Slope Nulls')
plt.violinplot(shuffled_slopes, positions=[2], showextrema=False, points=200, widths=[0.9])
plt.plot([1.75,2.25],[np.quantile(shuffled_slopes,0.025),np.quantile(shuffled_slopes,0.025)],'k:',linewidth=0.5)
plt.plot([1.75,2.25],[np.quantile(shuffled_slopes,0.975),np.quantile(shuffled_slopes,0.975)],'k:',linewidth=0.5)
x = plt.scatter(2, fn_cn_k, color=[0.2,0.2,0.2], s=20)
# set ticks
x.axes.get_xaxis().set_ticks([2])
x.axes.get_xaxis().set_ticklabels(['k2']);
# test for significance in a two-sided test
if np.logical_or(fn_cn_k > np.quantile(shuffled_slopes,0.975), fn_cn_k < np.quantile(shuffled_slopes,0.025)):
    plt.plot(2,0,'k*')

# save it
fname = '/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all cell summaries/within type flicker similarity connectome correlations.pdf'
flickdistfig.savefig(fname, format='pdf', orientation='portrait')
