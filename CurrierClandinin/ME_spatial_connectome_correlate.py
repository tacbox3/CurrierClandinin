# This script performs a variety of correlations between connectome morphology statistics and physiology data, including: within-type variance, TRF peak z-score, and RF center area

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

# save it
fname = '/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all cell summaries/RF area connectome correlations.pdf'
areafig.savefig(fname, format='pdf', orientation='landscape')
