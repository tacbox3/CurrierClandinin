# This script performs weighted summation modeling on both temporal filters and concatenated flicker response timecourses, as well as functional clustering and strong vs. weak input sampling analysis (i.e., all of Fig. 6)

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
from scipy import stats
from scipy.optimize import curve_fit

def line(x, k, b):
    y = k*x + b
    return (y)

def log(x, k, b):
    y = k*np.log(x) + b
    return (y)

def sigmoid(x, L , k, b):
    y = L / (1 + np.exp(-k*(x))) + b
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
book = xlrd.open_workbook('/Users/tcurrier/Desktop/Clandinin Lab/Connectomics/NernReiser2024 Data/HighN_InFracs_T4in_excluded.xls')
sheet = book.sheet_by_name('Nern 2024')
data = [[sheet.cell_value(r, c) for c in range(sheet.ncols)] for r in range(sheet.nrows)]
infrac_data = np.asarray(data)

# recover list of cell types present in the table
infrac_types = infrac_data[1,1:]
# replace empty entries with 0s
infrac_data[2:,1:][infrac_data[2:,1:] == ''] = '0'
# convert to floats
infrac_data = infrac_data[2:,1:].astype('float32')
# the connections data table indices are [input cell type, evaluated cell type], so to look at the fraction of input onto cell type A given by cell type B, you would access array element [B,A]

# manual entry of NT identity for cell types in infrac_types
infrac_NTs = np.asarray(['GABA','GABA','GABA','Glu','Glu','ACh','Glu','Glu','Glu','Glu','Glu','ACh','ACh','ACh','ACh','Glu','ACh','Glu','GABA','Glu','GABA','GABA','GABA','GABA','GABA','GABA','ACh','ACh','ACh','ACh','ACh','ACh','ACh','ACh','Glu','ACh','Glu','ACh','ACh','ACh','ACh','Glu','GABA','Glu','ACh','ACh','Glu','ACh','Glu'])

# for inhibitory input cell types, flip the sign of the connection weight
for ind in range(0,len(infrac_types)):
    if not infrac_NTs[ind] == 'ACh':
        infrac_data[ind,:] = -1 * infrac_data[ind,:]

# load pre-calculated temporal response data
all_cent_TRFs = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all_center_TRFs.npy')
all_surr_TRFs = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all_surround_TRFs.npy')
all_flicks = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all_flick_rois.npy')
pc_weights = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/full_PC_weights.npy')

# define flicker time vector and frequencies
flick_resamp_t = np.arange(0, 14-0.8, 1/5)
flick_freqs = [0.1,0.5,1,2]

#%% pre-compute summary metrics

# pull list of unique cell type labels in my data
utype_indices = np.unique(scraped_log[:,14]) != ''
unique_types = np.unique(scraped_log[:,14])[utype_indices]
Nbytype = np.zeros(len(unique_types))

# container arrays
mean_concat_TRFs = np.full((2*all_cent_TRFs.shape[0],len(unique_types)), np.nan, dtype='float')
mean_concat_surr_TRFs = np.full((2*all_surr_TRFs.shape[0],len(unique_types)), np.nan, dtype='float')
mean_concat_flicks = np.full((all_flicks.shape[0]*all_flicks.shape[1],len(unique_types)), np.nan, dtype='float')
mean_pc_coords = np.zeros((pc_weights.shape[1],len(unique_types)))

# create dictionaries to hold arrays of pairwise correlation coefficients for each cell type
TRF_pair_corrs = {}
surr_TRF_pair_corrs = {}
flick_pair_corrs = {}

# loop over cell types and populate physiology data arrays
for ind in range(0,len(unique_types)):
# for ind in range(54,55):
    # define cell type to summarize
    current_label = unique_types[ind]
    # define N for cell type
    current_cells = scraped_log[scraped_log[:,14] == current_label,1]
    current_n = len(current_cells)
    Nbytype[ind] = current_n
    # grab physiology data for current cell type
    label_cent_TRFs = all_cent_TRFs[:,:,scraped_log[:,14] == current_label]
    label_surr_TRFs = all_surr_TRFs[:,:,scraped_log[:,14] == current_label]
    label_flicks = all_flicks[:,:,scraped_log[:,14] == current_label]
    label_pc_coords = pc_weights[scraped_log[:,14] == current_label,:]
    # for TRFs, concatenate the two colors (axis=1), then calculate the within-type mean
    concat_TRFs = np.concatenate((label_cent_TRFs[:,0,:],label_cent_TRFs[:,1,:]),axis=0)
    mean_concat_TRFs[:,ind] = np.nanmean(concat_TRFs,axis=1)
    concat_surr_TRFs = np.concatenate((label_surr_TRFs[:,0,:],label_surr_TRFs[:,1,:]),axis=0)
    mean_concat_surr_TRFs[:,ind] = np.nanmean(concat_surr_TRFs,axis=1)
    # for flicker, concatenate the four directions (axis=0), then calculate the within-type mean
    concat_flicks = label_flicks.reshape((-1,label_flicks.shape[2]))
    mean_concat_flicks[:,ind] = np.nanmean(concat_flicks,axis=1)
    # for PCs, simply find the mean coordinates for the current cell type
    mean_pc_coords[:,ind] = np.nanmean(label_pc_coords, axis=0)

    # additionally, calculate the within-type cross-correlation distribution as a measure of variability (this should just be saved and passed to the spatial code, where it can be used exactly like the PC space pairwise distance metric).
    nl_concat_TRFs = np.zeros((concat_TRFs.shape[0],1))
    nl_concat_surr_TRFs = np.zeros((concat_surr_TRFs.shape[0],1))
    for cell in range(0,concat_TRFs.shape[1]):
        if not np.any(np.isnan(concat_TRFs[:,cell])):
            # remove cells containing nans (i.e. data from one color is missing)
            nl_concat_TRFs = np.append(nl_concat_TRFs, concat_TRFs[:,cell].reshape((-1,1)), axis=1)
            nl_concat_surr_TRFs = np.append(nl_concat_surr_TRFs, concat_surr_TRFs[:,cell].reshape((-1,1)), axis=1)
    # remove intialization row
    nl_concat_TRFs = nl_concat_TRFs[:,1:]
    nl_concat_surr_TRFs = nl_concat_surr_TRFs[:,1:]
    # take correlation coefficients, zeroing out duplicate entries on and above the main diagonal, then save to dictionary with a key corresponding to the cell type name
    if nl_concat_TRFs.shape[1] > 1:
        cmat = np.tril(np.corrcoef(nl_concat_TRFs, rowvar=False), k=-1)
        TRF_pair_corrs[current_label] = cmat[np.nonzero(cmat)]
        cmat = np.tril(np.corrcoef(nl_concat_surr_TRFs, rowvar=False), k=-1)
        surr_TRF_pair_corrs[current_label] = cmat[np.nonzero(cmat)]

    # flicker cross-correlations
    nl_concat_flicks = np.zeros((concat_flicks.shape[0],1))
    for cell in range(0,concat_flicks.shape[1]):
        if not np.any(np.isnan(concat_flicks[:,cell])):
            # remove cells containing nans (i.e. data from one color is missing)
            nl_concat_flicks = np.append(nl_concat_flicks, concat_flicks[:,cell].reshape((-1,1)), axis=1)
    # remove intialization row
    nl_concat_flicks = nl_concat_flicks[:,1:]
    # take correlation coefficients, zeroing out duplicate entries on and above the main diagonal, then save to dictionary with a key corresponding to the cell type name
    if nl_concat_flicks.shape[1] > 1:
        cmat = np.tril(np.corrcoef(nl_concat_flicks, rowvar=False), k=-1)
        flick_pair_corrs[current_label] = cmat[np.nonzero(cmat)]

#%% construct mean-vs-mean correlation table for all cell type pairs in the infrac_types list

# define threshold on infrac (discards connections below this level). a 1% higher threshold will cut off approximately 1-4 additional synapses, depending on the cell type. a threshold of 2 is the closest match to the "standard" 5 synapse cutoff
infrac_thresh = 2

# container arrays for cross-type correlations of mean temporal responses
cTRF_corrs_table = np.zeros((len(infrac_types),len(infrac_types)))
sTRF_corrs_table = np.zeros((len(infrac_types),len(infrac_types)))
flick_corrs_table = np.zeros((len(infrac_types),len(infrac_types)))
pc_dists_table = np.zeros((len(infrac_types),len(infrac_types)))

# populate tables one pair at a time
for ind1 in range(0,len(infrac_types)):
    first_label = infrac_types[ind1]
    cTRF1 = mean_concat_TRFs[:,unique_types == first_label]
    sTRF1 = mean_concat_surr_TRFs[:,unique_types == first_label]
    flick1 = mean_concat_flicks[:,unique_types == first_label]
    pc1 = mean_pc_coords[:,unique_types == first_label]
    for ind2 in range(ind1+1,len(infrac_types)):
        second_label = infrac_types[ind2]
        cTRF2 = mean_concat_TRFs[:,unique_types == second_label]
        sTRF2 = mean_concat_surr_TRFs[:,unique_types == second_label]
        flick2 = mean_concat_flicks[:,unique_types == second_label]
        pc2 = mean_pc_coords[:,unique_types == second_label]
        # build corrcoef input arrays
        cTRFs = np.append(cTRF1.reshape(-1,1), cTRF2.reshape(-1,1), axis=1)
        sTRFs = np.append(sTRF1.reshape(-1,1), sTRF2.reshape(-1,1), axis=1)
        flicks_array = np.append(flick1.reshape(-1,1), flick2.reshape(-1,1), axis=1)
        # perform correlations
        cTRF_corrs_table[ind1,ind2] = np.corrcoef(cTRFs, rowvar=False)[0,1]
        sTRF_corrs_table[ind1,ind2] = np.corrcoef(sTRFs, rowvar=False)[0,1]
        flick_corrs_table[ind1,ind2] = np.corrcoef(flicks_array, rowvar=False)[0,1]
        # calculate PC space distance
        pc_dists_table[ind1,ind2] = np.linalg.norm(pc1-pc2)

# reflect across the main diagonal to make table lookup easier in the weighting steps
lower_inds = np.nonzero(np.tril(np.ones((len(infrac_types),len(infrac_types)))))
cTRF_corrs_table[lower_inds] = cTRF_corrs_table.T[lower_inds]
sTRF_corrs_table[lower_inds] = sTRF_corrs_table.T[lower_inds]
flick_corrs_table[lower_inds] = flick_corrs_table.T[lower_inds]
pc_dists_table[lower_inds] = pc_dists_table.T[lower_inds]

# fill diagonals with NaNs to exclude weighting self-connections
np.fill_diagonal(cTRF_corrs_table, np.nan)
np.fill_diagonal(sTRF_corrs_table, np.nan)
np.fill_diagonal(flick_corrs_table, np.nan)
np.fill_diagonal(pc_dists_table, np.nan)

# define vectorized versions for plotting, taking only indices with non-zero input fractions
infrac_vec = infrac_data.flatten()[np.nonzero(infrac_data.flatten())]
cTRF_vec = cTRF_corrs_table.flatten()[np.nonzero(infrac_data.flatten())]
sTRF_vec = sTRF_corrs_table.flatten()[np.nonzero(infrac_data.flatten())]
flick_vec = flick_corrs_table.flatten()[np.nonzero(infrac_data.flatten())]
pc_vec = pc_dists_table.flatten()[np.nonzero(infrac_data.flatten())]
unused_vec = cTRF_corrs_table.flatten()[infrac_data.flatten()==0]
flick_unused_vec = flick_corrs_table.flatten()[infrac_data.flatten()==0]

# thresholded versions at infrac_thresh % or more
# convert sub-threshold connections to 0s
thresh_infrac_vec = np.where(np.abs(infrac_vec)>=infrac_thresh,infrac_vec,0)
# retain indices of non-zero infracs in thresholded vectors
thresh_cTRF_vec = cTRF_vec[np.nonzero(thresh_infrac_vec)]
thresh_sTRF_vec = sTRF_vec[np.nonzero(thresh_infrac_vec)]
thresh_flick_vec = flick_vec[np.nonzero(thresh_infrac_vec)]
thresh_pc_vec = pc_vec[np.nonzero(thresh_infrac_vec)]
thresh_infrac_vec = thresh_infrac_vec[np.nonzero(thresh_infrac_vec)]

#%% plotting
tempsimfig = plt.figure(figsize=(18, 16))
numshuffles = 10000

# TRF CENTER CORR VS. INFRAC
plt.subplot(3,3,1)
fn_var = thresh_cTRF_vec[thresh_infrac_vec > 0]
cn_var = thresh_infrac_vec[thresh_infrac_vec > 0]
# remove nans
nl_cn_var = cn_var[~np.isnan(fn_var)]
nl_fn_var = fn_var[~np.isnan(fn_var)]
# correlation coefficient
corrmat = np.corrcoef(np.asarray([nl_cn_var,nl_fn_var]))
fn_cn_R = np.round(corrmat[0,1],2)
# linear fit
xdata = nl_cn_var
ydata = nl_fn_var
p0 = [10, 0] #initial conditions for gradient descent
popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
fn_cn_k = popt[0]
# scatter data
plt.scatter(nl_cn_var,nl_fn_var,color=[1,0,0],s=5)
plt.plot(np.arange(0,75,1),np.arange(0,75,1)*fn_cn_k+popt[1],'r-')
plt.title('R = ' + str(fn_cn_R) + ', k = ' + str(np.round(fn_cn_k,5)) + ' /%')
plt.ylabel('Center TRF 0-lag Cross-Correlation')
plt.xlabel('Input Fraction (%)')
plt.ylim([-1.05,1.05])
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
plt.plot([0.75,1.25],[np.quantile(shuffled_corrs,0.025),np.quantile(shuffled_corrs,0.025)],'k:',linewidth=0.5)
plt.plot([0.75,1.25],[np.quantile(shuffled_corrs,0.975),np.quantile(shuffled_corrs,0.975)],'k:',linewidth=0.5)
plt.scatter(1, fn_cn_R, color=[0.2,0.2,0.2], s=20)
# test for significance in a two-sided test
if np.logical_or(fn_cn_R > np.quantile(shuffled_corrs,0.975), fn_cn_R < np.quantile(shuffled_corrs,0.025)):
    plt.plot(1,0,'k*')
# plot slopes
plt.subplot(3,6,6)
plt.title('Slope Nulls')
plt.violinplot(shuffled_slopes, positions=[1], showextrema=False, points=200, widths=[0.9])
plt.plot([0.75,1.25],[np.quantile(shuffled_slopes,0.025),np.quantile(shuffled_slopes,0.025)],'k:',linewidth=0.5)
plt.plot([0.75,1.25],[np.quantile(shuffled_slopes,0.975),np.quantile(shuffled_slopes,0.975)],'k:',linewidth=0.5)
plt.scatter(1, fn_cn_k, color=[0.2,0.2,0.2], s=20)
# test for significance in a two-sided test
if np.logical_or(fn_cn_k > np.quantile(shuffled_slopes,0.975), fn_cn_k < np.quantile(shuffled_slopes,0.025)):
    plt.plot(1,0,'k*')

# TRF SURROUND CORR VS. INFRAC
plt.subplot(3,3,2)
fn_var = thresh_sTRF_vec[thresh_infrac_vec > 0]
cn_var = thresh_infrac_vec[thresh_infrac_vec > 0]
# remove nans
nl_cn_var = cn_var[~np.isnan(fn_var)]
nl_fn_var = fn_var[~np.isnan(fn_var)]
# correlation coefficient
corrmat = np.corrcoef(np.asarray([nl_cn_var,nl_fn_var]))
fn_cn_R = np.round(corrmat[0,1],2)
# linear fit
xdata = nl_cn_var
ydata = nl_fn_var
p0 = [10, 0] #initial conditions for gradient descent
popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
fn_cn_k = popt[0]
# scatter data
plt.scatter(nl_cn_var,nl_fn_var,color=[1,0,0],s=5)
plt.plot(np.arange(0,75,1),np.arange(0,75,1)*fn_cn_k+popt[1],'r-')
plt.title('R = ' + str(fn_cn_R) + ', k = ' + str(np.round(fn_cn_k,5)) + ' /%')
plt.ylabel('Surround TRF 0-lag Cross-Correlation')
plt.xlabel('Input Fraction (%)')
plt.ylim([-1.05,1.05])
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
plt.subplot(3,6,6)
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

# TRF CENTER CORR VS. INFRAC
plt.subplot(3,3,1)
fn_var = thresh_cTRF_vec[thresh_infrac_vec < 0]
cn_var = thresh_infrac_vec[thresh_infrac_vec < 0]
# remove nans
nl_cn_var = cn_var[~np.isnan(fn_var)]
nl_fn_var = fn_var[~np.isnan(fn_var)]
# correlation coefficient
corrmat = np.corrcoef(np.asarray([nl_cn_var,nl_fn_var]))
fn_cn_R = np.round(corrmat[0,1],2)
# linear fit
xdata = nl_cn_var
ydata = nl_fn_var
p0 = [10, 2] #initial conditions for gradient descent
popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
fn_cn_k = popt[0]
# scatter data
plt.scatter(nl_cn_var,nl_fn_var,color=[0,0,1],s=5)
plt.plot(np.arange(-45,0,1),np.arange(-45,0,1)*fn_cn_k+popt[1],'b-')
plt.title('R = ' + str(fn_cn_R) + ', k = ' + str(np.round(fn_cn_k,5)) + ' /%')
plt.ylabel('Center TRF 0-lag Cross-Correlation')
plt.xlabel('Input Fraction (%)')
plt.ylim([-1.05,1.05])
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
plt.violinplot(shuffled_corrs, positions=[3], showextrema=False, points=200, widths=[0.9])
plt.plot([2.75,3.25],[np.quantile(shuffled_corrs,0.025),np.quantile(shuffled_corrs,0.025)],'k:',linewidth=0.5)
plt.plot([2.75,3.25],[np.quantile(shuffled_corrs,0.975),np.quantile(shuffled_corrs,0.975)],'k:',linewidth=0.5)
plt.scatter(3, fn_cn_R, color=[0.2,0.2,0.2], s=20)
# test for significance in a two-sided test
if np.logical_or(fn_cn_R > np.quantile(shuffled_corrs,0.975), fn_cn_R < np.quantile(shuffled_corrs,0.025)):
    plt.plot(3,0,'k*')
# plot slopes
plt.subplot(3,6,6)
plt.title('Slope Nulls')
plt.violinplot(shuffled_slopes, positions=[3], showextrema=False, points=200, widths=[0.9])
plt.plot([2.75,3.25],[np.quantile(shuffled_slopes,0.025),np.quantile(shuffled_slopes,0.025)],'k:',linewidth=0.5)
plt.plot([2.75,3.25],[np.quantile(shuffled_slopes,0.975),np.quantile(shuffled_slopes,0.975)],'k:',linewidth=0.5)
plt.scatter(3, fn_cn_k, color=[0.2,0.2,0.2], s=20)
# test for significance in a two-sided test
if np.logical_or(fn_cn_k > np.quantile(shuffled_slopes,0.975), fn_cn_k < np.quantile(shuffled_slopes,0.025)):
    plt.plot(3,0,'k*')

# TRF SURROUND CORR VS. INFRAC
plt.subplot(3,3,2)
fn_var = thresh_sTRF_vec[thresh_infrac_vec < 0]
cn_var = thresh_infrac_vec[thresh_infrac_vec < 0]
# remove nans
nl_cn_var = cn_var[~np.isnan(fn_var)]
nl_fn_var = fn_var[~np.isnan(fn_var)]
# correlation coefficient
corrmat = np.corrcoef(np.asarray([nl_cn_var,nl_fn_var]))
fn_cn_R = np.round(corrmat[0,1],2)
# linear fit
xdata = nl_cn_var
ydata = nl_fn_var
p0 = [10, 2] #initial conditions for gradient descent
popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
fn_cn_k = popt[0]
# scatter data
plt.scatter(nl_cn_var,nl_fn_var,color=[0,0,1],s=5)
plt.plot(np.arange(-45,0,1),np.arange(-45,0,1)*fn_cn_k+popt[1],'b-')
plt.title('R = ' + str(fn_cn_R) + ', k = ' + str(np.round(fn_cn_k,5)) + ' /%')
plt.ylabel('Surround TRF 0-lag Cross-Correlation')
plt.xlabel('Input Fraction (%)')
plt.ylim([-1.05,1.05])
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
plt.violinplot(shuffled_corrs, positions=[4], showextrema=False, points=200, widths=[0.9])
plt.plot([3.75,4.25],[np.quantile(shuffled_corrs,0.025),np.quantile(shuffled_corrs,0.025)],'k:',linewidth=0.5)
plt.plot([3.75,4.25],[np.quantile(shuffled_corrs,0.975),np.quantile(shuffled_corrs,0.975)],'k:',linewidth=0.5)
x = plt.scatter(4, fn_cn_R, color=[0.2,0.2,0.2], s=20)
# set ticks
x.axes.get_xaxis().set_ticks([1,2,3,4])
x.axes.get_xaxis().set_ticklabels(['ER1','ER2','IR1','IR2']);
# test for significance in a two-sided test
if np.logical_or(fn_cn_R > np.quantile(shuffled_corrs,0.975), fn_cn_R < np.quantile(shuffled_corrs,0.025)):
    plt.plot(4,0,'k*')
# plot slopes
plt.subplot(3,6,6)
plt.title('Slope Nulls')
plt.violinplot(shuffled_slopes, positions=[4], showextrema=False, points=200, widths=[0.9])
plt.plot([3.75,4.25],[np.quantile(shuffled_slopes,0.025),np.quantile(shuffled_slopes,0.025)],'k:',linewidth=0.5)
plt.plot([3.75,4.25],[np.quantile(shuffled_slopes,0.975),np.quantile(shuffled_slopes,0.975)],'k:',linewidth=0.5)
x = plt.scatter(4, fn_cn_k, color=[0.2,0.2,0.2], s=20)
# set ticks
x.axes.get_xaxis().set_ticks([1,2,3,4])
x.axes.get_xaxis().set_ticklabels(['Ek1','Ek2','Ik1','Ik2']);
# test for significance in a two-sided test
if np.logical_or(fn_cn_k > np.quantile(shuffled_slopes,0.975), fn_cn_k < np.quantile(shuffled_slopes,0.025)):
    plt.plot(4,0,'k*')


# PLOT CONNECTION MATRIX
plt.subplot(3,3,4)
plt.title('Connectivity Matrix')
plt.ylabel('Presynaptic Cell')
plt.xlabel('Postsynaptic Cell')
x = plt.imshow(infrac_data, cmap='bwr', clim=[-75,75])
# blank ticks
x.axes.get_xaxis().set_ticks([])
x.axes.get_yaxis().set_ticks([])
plt.colorbar(ticks=[-75,-50,-25,0,25,50,75],)


# FLICKER CORR VS. INFRAC
plt.subplot(3,3,5)
fn_var = thresh_flick_vec[thresh_infrac_vec > 0]
cn_var = thresh_infrac_vec[thresh_infrac_vec > 0]
# remove nans
nl_cn_var = cn_var[~np.isnan(fn_var)]
nl_fn_var = fn_var[~np.isnan(fn_var)]
# correlation coefficient
corrmat = np.corrcoef(np.asarray([nl_cn_var,nl_fn_var]))
fn_cn_R = np.round(corrmat[0,1],2)
# linear fit
xdata = nl_cn_var
ydata = nl_fn_var
p0 = [10, 0] #initial conditions for gradient descent
popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
fn_cn_k = popt[0]
# scatter data
plt.scatter(nl_cn_var,nl_fn_var,color=[1,0,0],s=5)
plt.plot(np.arange(0,75,1),np.arange(0,75,1)*fn_cn_k+popt[1],'r-')
plt.title('R = ' + str(fn_cn_R) + ', k = ' + str(np.round(fn_cn_k,5)) + ' /%')
plt.ylabel('Flicker 0-lag Cross-Correlation')
plt.xlabel('Input Fraction (%)')
plt.ylim([-1.05,1.05])
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
x = plt.scatter(1, fn_cn_R, color=[0.2,0.2,0.2], s=20)
# test for significance in a two-sided test
if np.logical_or(fn_cn_R > np.quantile(shuffled_corrs,0.975), fn_cn_R < np.quantile(shuffled_corrs,0.025)):
    plt.plot(1,0,'k*')
# plot slopes
plt.subplot(3,6,12)
plt.title('Slope Nulls')
plt.violinplot(shuffled_slopes, positions=[1], showextrema=False, points=200, widths=[0.9])
plt.plot([0.75,1.25],[np.quantile(shuffled_slopes,0.025),np.quantile(shuffled_slopes,0.025)],'k:',linewidth=0.5)
plt.plot([0.75,1.25],[np.quantile(shuffled_slopes,0.975),np.quantile(shuffled_slopes,0.975)],'k:',linewidth=0.5)
x = plt.scatter(1, fn_cn_k, color=[0.2,0.2,0.2], s=20)
# test for significance in a two-sided test
if np.logical_or(fn_cn_k > np.quantile(shuffled_slopes,0.975), fn_cn_k < np.quantile(shuffled_slopes,0.025)):
    plt.plot(1,0,'k*')

# FLICKER CORR VS. INFRAC
plt.subplot(3,3,5)
fn_var = thresh_flick_vec[thresh_infrac_vec < 0]
cn_var = thresh_infrac_vec[thresh_infrac_vec < 0]
# remove nans
nl_cn_var = cn_var[~np.isnan(fn_var)]
nl_fn_var = fn_var[~np.isnan(fn_var)]
# correlation coefficient
corrmat = np.corrcoef(np.asarray([nl_cn_var,nl_fn_var]))
fn_cn_R = np.round(corrmat[0,1],2)
# linear fit
xdata = nl_cn_var
ydata = nl_fn_var
p0 = [10, 0] #initial conditions for gradient descent
popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
fn_cn_k = popt[0]
# scatter data
plt.scatter(nl_cn_var,nl_fn_var,color=[0,0,1],s=5)
plt.plot(np.arange(-45,0,1),np.arange(-45,0,1)*fn_cn_k+popt[1],'b-')
plt.title('R = ' + str(fn_cn_R) + ', k = ' + str(np.round(fn_cn_k,5)) + ' /%')
plt.ylabel('Flicker 0-lag Cross-Correlation')
plt.xlabel('Input Fraction (%)')
plt.ylim([-1.05,1.05])
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
x.axes.get_xaxis().set_ticklabels(['ER1','IR1']);
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
x.axes.get_xaxis().set_ticklabels(['Ek1','Ik1']);
# test for significance in a two-sided test
if np.logical_or(fn_cn_k > np.quantile(shuffled_slopes,0.975), fn_cn_k < np.quantile(shuffled_slopes,0.025)):
    plt.plot(2,0,'k*')


# PC DIST VS. INFRAC
plt.subplot(3,3,8)
fn_var = thresh_pc_vec[thresh_infrac_vec > 0]
cn_var = thresh_infrac_vec[thresh_infrac_vec > 0]
# remove nans
nl_cn_var = cn_var[~np.isnan(fn_var)]
nl_fn_var = fn_var[~np.isnan(fn_var)]
# correlation coefficient
corrmat = np.corrcoef(np.asarray([nl_cn_var,nl_fn_var]))
fn_cn_R = np.round(corrmat[0,1],2)
# linear fit
xdata = nl_cn_var
ydata = nl_fn_var
p0 = [10, 0] #initial conditions for gradient descent
popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
fn_cn_k = popt[0]
# scatter data
plt.scatter(nl_cn_var,nl_fn_var,color=[1,0,0],s=5)
plt.plot(np.arange(0,75,1),np.arange(0,75,1)*fn_cn_k+popt[1],'r-')
plt.title('R = ' + str(fn_cn_R) + ', k = ' + str(np.round(fn_cn_k,5)) + ' /%')
plt.ylabel('Pairwise Distance in PC Space')
plt.xlabel('Input Fraction (%)')
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
plt.plot([0.75,1.25],[np.quantile(shuffled_corrs,0.025),np.quantile(shuffled_corrs,0.025)],'k:',linewidth=0.5)
plt.plot([0.75,1.25],[np.quantile(shuffled_corrs,0.975),np.quantile(shuffled_corrs,0.975)],'k:',linewidth=0.5)
x = plt.scatter(1, fn_cn_R, color=[0.2,0.2,0.2], s=20)
# test for significance in a two-sided test
if np.logical_or(fn_cn_R > np.quantile(shuffled_corrs,0.975), fn_cn_R < np.quantile(shuffled_corrs,0.025)):
    plt.plot(1,0,'k*')
# plot slopes
plt.subplot(3,6,18)
plt.title('Slope Nulls')
plt.violinplot(shuffled_slopes, positions=[1], showextrema=False, points=200, widths=[0.9])
plt.plot([0.75,1.25],[np.quantile(shuffled_slopes,0.025),np.quantile(shuffled_slopes,0.025)],'k:',linewidth=0.5)
plt.plot([0.75,1.25],[np.quantile(shuffled_slopes,0.975),np.quantile(shuffled_slopes,0.975)],'k:',linewidth=0.5)
x = plt.scatter(1, fn_cn_k, color=[0.2,0.2,0.2], s=20)
# test for significance in a two-sided test
if np.logical_or(fn_cn_k > np.quantile(shuffled_slopes,0.975), fn_cn_k < np.quantile(shuffled_slopes,0.025)):
    plt.plot(1,0,'k*')

# PC DISTANCE VS. INFRAC
plt.subplot(3,3,8)
fn_var = thresh_pc_vec[thresh_infrac_vec < 0]
cn_var = thresh_infrac_vec[thresh_infrac_vec < 0]
# remove nans
nl_cn_var = cn_var[~np.isnan(fn_var)]
nl_fn_var = fn_var[~np.isnan(fn_var)]
# correlation coefficient
corrmat = np.corrcoef(np.asarray([nl_cn_var,nl_fn_var]))
fn_cn_R = np.round(corrmat[0,1],2)
# linear fit
xdata = nl_cn_var
ydata = nl_fn_var
p0 = [10, 0] #initial conditions for gradient descent
popt, pcov = curve_fit(line, xdata, ydata, p0, method='lm')
fn_cn_k = popt[0]
# scatter data
plt.scatter(nl_cn_var,nl_fn_var,color=[0,0,1],s=5)
plt.plot(np.arange(-45,0,1),np.arange(-45,0,1)*fn_cn_k+popt[1],'b-')
plt.title('R = ' + str(fn_cn_R) + ', k = ' + str(np.round(fn_cn_k,5)) + ' /%')
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
x.axes.get_xaxis().set_ticks([1,2])
x.axes.get_xaxis().set_ticklabels(['ER1','IR1']);
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
x.axes.get_xaxis().set_ticks([1,2])
x.axes.get_xaxis().set_ticklabels(['Ek1','Ik1']);
# test for significance in a two-sided test
if np.logical_or(fn_cn_k > np.quantile(shuffled_slopes,0.975), fn_cn_k < np.quantile(shuffled_slopes,0.025)):
    plt.plot(2,0,'k*')


# save it
fname = '/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all cell summaries/temporal similarity vs connection weight.pdf'
tempsimfig.savefig(fname, format='pdf', orientation='landscape')

#%% for each postsynaptic cell type, plot cross-correlation vs. cumulative weighted input fraction as inputs are added, from strongest to weakest
tempcumfig = plt.figure(figsize=(24, 18))
binwidths = 20 # width of bins in units of input %

# construct container for appending binned Rs
binned_Rs = np.full((int(100/binwidths),1), np.nan, dtype='float')
binned_flick_Rs = np.full((int(100/binwidths),1), np.nan, dtype='float')

# container arrays
first_input_infrac = np.zeros(len(infrac_types))
first_input_corrs = np.zeros(len(infrac_types))
last_input_corrs = np.zeros(len(infrac_types))
nobig_input_corrs = np.zeros(len(infrac_types))
peak_corrs = np.zeros(len(infrac_types))
infrac_at_pk_corrs = np.zeros(len(infrac_types))
flick_first_input_corrs = np.zeros(len(infrac_types))
flick_last_input_corrs = np.zeros(len(infrac_types))
flick_nobig_input_corrs = np.zeros(len(infrac_types))
flick_peak_corrs = np.zeros(len(infrac_types))
flick_infrac_at_pk_corrs = np.zeros(len(infrac_types))
total_infracs = np.zeros(len(infrac_types))

# for post_ind in range(29,30):
for post_ind in range(0,len(infrac_types)):
    # grab current cell type and data
    post_label = infrac_types[post_ind]
    post_TRF = mean_concat_TRFs[:,unique_types == post_label]
    post_flicks = mean_concat_flicks[:,unique_types == post_label]
    # grab input weights for current cell type
    input_weights = infrac_data[:,post_ind]
    # rank inputs by weight
    input_ranks = np.flip(np.argsort(np.abs(input_weights)))
    ranked_inputs = input_weights[input_ranks]
    ranked_types = infrac_types[input_ranks]
    # cut off 0-weight inputs
    ranked_types = ranked_types[np.nonzero(ranked_inputs)]
    ranked_inputs = ranked_inputs[np.nonzero(ranked_inputs)]
    # take the cumulative sum of the absolute value of input fractions
    cumsum_infracs = np.cumsum(np.abs(ranked_inputs))
    # container array for input data
    input_TRFs = np.zeros((mean_concat_TRFs.shape[0],len(ranked_inputs)))
    input_flicks = np.zeros((mean_concat_flicks.shape[0],len(ranked_inputs)))
    # collect data for input cell types
    for pre_ind in range(0,len(ranked_inputs)):
        current_label = ranked_types[pre_ind]
        input_TRFs[:,pre_ind] = mean_concat_TRFs[:,unique_types == current_label].reshape(mean_concat_TRFs.shape[0]) #idk why it needs a reshape here
        input_flicks[:,pre_ind] = mean_concat_flicks[:,unique_types == current_label].reshape(mean_concat_flicks.shape[0])
    # weight the data by the input fraction
    weighted_input_TRFs = input_TRFs*ranked_inputs/100
    weighted_input_flicks = input_flicks*ranked_inputs/100
    # cumulatively sum over dimension 1 to yield a new TRF/flicker response for each added input
    cumsum_input_TRFs = np.cumsum(weighted_input_TRFs, axis=1)
    cumsum_input_flicks = np.cumsum(weighted_input_flicks, axis=1)
    nobig_cumsum_input_TRFs = np.cumsum(weighted_input_TRFs[:,1:], axis=1)
    nobig_cumsum_input_flicks = np.cumsum(weighted_input_flicks[:,1:], axis=1)
    # take the cross-correlation between the post-synaptic cell and each of the cumulative TRFs/flicks
    corr_list = np.corrcoef(np.append(post_TRF, cumsum_input_TRFs, axis=1), rowvar=False)[0,1:]
    flick_corr_list = np.corrcoef(np.append(post_flicks, cumsum_input_flicks, axis=1), rowvar=False)[0,1:]
    nobig_corr_list = np.corrcoef(np.append(post_TRF, nobig_cumsum_input_TRFs, axis=1), rowvar=False)[0,1:]
    nobig_flick_corr_list = np.corrcoef(np.append(post_flicks, nobig_cumsum_input_flicks, axis=1), rowvar=False)[0,1:]
    # define bins to place corr data for each successive input
    cumsum_bins = (cumsum_infracs//binwidths).astype('int')
    # loop over ranked inputs, dropping R or distance data into appropriate cum_infrac bins
    for input in range(0,len(cumsum_infracs)):
        # create a nan vector of correct length, then change correct bin's value to corrcoef
        binvec = np.full((int(100/binwidths),1), np.nan, dtype='float')
        binvec[cumsum_bins[input]] = corr_list[input]
        # append this binvec to binned_Rs
        binned_Rs = np.append(binned_Rs, binvec, axis=1)
        # repeat for flicker
        binvec = np.full((int(100/binwidths),1), np.nan, dtype='float')
        binvec[cumsum_bins[input]] = flick_corr_list[input]
        binned_flick_Rs = np.append(binned_flick_Rs, binvec, axis=1)
    # save summary metrics for current cell type: first input corr, last input cumcorr, peak corr vs. input cumsum
    first_input_corrs[post_ind] = corr_list[0]
    last_input_corrs[post_ind] = corr_list[len(corr_list)-1]
    nobig_input_corrs[post_ind] = nobig_corr_list[len(nobig_corr_list)-1]
    peak_corrs[post_ind] = corr_list[np.argmax(corr_list)]
    infrac_at_pk_corrs[post_ind] = cumsum_infracs[np.argmax(corr_list)]
    first_input_infrac[post_ind] = cumsum_infracs[0]
    flick_first_input_corrs[post_ind] = flick_corr_list[0]
    flick_last_input_corrs[post_ind] = flick_corr_list[len(flick_corr_list)-1]
    flick_nobig_input_corrs[post_ind] = nobig_flick_corr_list[len(nobig_flick_corr_list)-1]
    flick_peak_corrs[post_ind] = flick_corr_list[np.argmax(flick_corr_list)]
    flick_infrac_at_pk_corrs[post_ind] = cumsum_infracs[np.argmax(flick_corr_list)]
    total_infracs[post_ind] = np.max(cumsum_infracs)

    # plot correlation by input fraction sum
    plt.subplot(3,4,1)
    plt.plot(cumsum_infracs, corr_list, linewidth=0.3, color=[0.3,0.3,0.3])
    plt.subplot(3,4,5)
    plt.plot(cumsum_infracs, flick_corr_list, linewidth=0.3, color=[0.3,0.3,0.3])


# TRF plot props
plt.subplot(3,4,1)
plt.plot(np.arange(binwidths/2,100+binwidths/2,binwidths), np.nanmedian(binned_Rs, axis=1), '-', linewidth=2, color=[0.3,0.6,1])
plt.ylabel('Similarity to Post-Synaptic TRF (R)')
plt.xlabel('Cumulative Input Fraction')
plt.title('Connection-Weighted Sum of Pre-Synaptic TRFs')
plt.xlim(0,100)
plt.ylim(-1.1,1.1)
plt.subplot(3,4,2)
plt.plot([20,20],[-1,1],':',color=[0.5,0.5,0.5])
plt.scatter(first_input_infrac, first_input_corrs, s=20, color=[0.2,0.2,0.2])
# correlation coefficient
corrmat = np.corrcoef(np.asarray([first_input_infrac,first_input_corrs]))
fn_cn_R = np.round(corrmat[0,1],2)
plt.text(50,-0.7,'R = ' + str(fn_cn_R))
# shuffle with replacement
fn_var = first_input_corrs
cn_var = first_input_infrac
# remove nans
nl_cn_var = cn_var[~np.isnan(fn_var)]
nl_fn_var = fn_var[~np.isnan(fn_var)]
shuffs_fn_var = np.random.choice(nl_fn_var, size=(len(nl_fn_var),10000), replace=True)
# calculate correlation coefficient distribution
shuffled_corrs = np.corrcoef((np.append(nl_cn_var.reshape(-1,1), shuffs_fn_var, axis=1)).T)[0,1:]
# plot correlation coefficients
plt.subplot(3,4,3)
plt.violinplot(shuffled_corrs, positions=[4], showextrema=False, points=200, widths=[0.9])
plt.plot([3.75,4.25],[np.quantile(shuffled_corrs,0.025),np.quantile(shuffled_corrs,0.025)],'k:',linewidth=0.5)
plt.plot([3.75,4.25],[np.quantile(shuffled_corrs,0.975),np.quantile(shuffled_corrs,0.975)],'k:',linewidth=0.5)
x = plt.scatter(4, fn_cn_R, color=[0.2,0.2,0.2], s=20)
# set ticks
x.axes.get_xaxis().set_ticks([4])
x.axes.get_xaxis().set_ticklabels(['Corr. Coef. Null']);
# test for significance in a two-sided test
if np.logical_or(fn_cn_R > np.quantile(shuffled_corrs,0.975), fn_cn_R < np.quantile(shuffled_corrs,0.025)):
    plt.plot(4,0,'k*')
# plot binned means
plt.subplot(3,4,2)
pk_bins = np.digitize(first_input_infrac, bins=np.arange(15/2,75+15/2,15))
pk_corr_binned_means = np.zeros(len(np.arange(15/2,75+15/2,15)))
for bin in range(0,len(np.arange(15/2,75+15/2,15))):
    pk_corr_binned_means[bin] = np.median(first_input_corrs[pk_bins==bin])
# manually fill in empty bin with mean of prior and latter
pk_corr_binned_means[3] = (pk_corr_binned_means[2]+pk_corr_binned_means[4])/2
plt.plot(np.arange(15/2,75,15), pk_corr_binned_means, '-', linewidth=2, color=[0.3,0.6,1])
plt.xlabel('Input Fraction of Largest Input (%)')
plt.title('Correlation vs. Input Fraction for Largest Input')
plt.xlim(0,80)
plt.ylim(-1.1,1.1)
plt.subplot(3,4,3)
for ind in range(0,len(infrac_types)):
    plt.plot([1,2,3],[first_input_corrs[ind],last_input_corrs[ind],nobig_input_corrs[ind]],'.-',linewidth=0.5,color=[0.5,0.5,0.5])
plt.plot([1,2,3],[np.median(first_input_corrs),np.median(last_input_corrs),np.median(nobig_input_corrs)],'.-', linewidth=2,color=[0.3,0.6,1])
# signed-rank test against null that distributions are the same
statistic, pvalue = stats.wilcoxon(first_input_corrs, y=last_input_corrs)
plt.text(1.15, 1.02, 'p = ' + str(np.round(pvalue,3)))
statistic, pvalue = stats.wilcoxon(last_input_corrs, y=nobig_input_corrs)
plt.text(2.15, 1.02, 'p = ' + str(np.round(pvalue,3)))
statistic, pvalue = stats.wilcoxon(first_input_corrs, y=nobig_input_corrs)
plt.text(1.65, -1.08, 'p = ' + str(np.round(pvalue,3)))
plt.title('Largest Input vs. All Inputs vs. All-Largest')
plt.xlim(0.5,4.5)
plt.ylim(-1.1,1.1)

plt.subplot(3,4,4)
# define plot groups (weak vs. strong, E vs. I), multiply I groups by -1 to "sign correct" correlations
weakE = cTRF_vec[np.logical_and(infrac_vec < infrac_thresh, infrac_vec > 0)]
strongE = thresh_cTRF_vec[thresh_infrac_vec > 0]
weakI = cTRF_vec[np.logical_and(infrac_vec > -1*infrac_thresh, infrac_vec < 0)]
strongI = thresh_cTRF_vec[thresh_infrac_vec < 0]
allEI = cTRF_vec[np.nonzero(infrac_vec)]
# remove nans from each plot group
nl_weakE = weakE[~np.isnan(weakE)]
nl_strongE = strongE[~np.isnan(strongE)]
nl_weakI = weakI[~np.isnan(weakI)]
nl_strongI = strongI[~np.isnan(strongI)]
nl_allEI = allEI[~np.isnan(allEI)]
nl_unused = unused_vec[~np.isnan(unused_vec)]
# plot violins of each distribution
plt.violinplot(nl_weakI, positions=[1], showextrema=False, points=200, widths=[0.9])
plt.violinplot(nl_weakE, positions=[2], showextrema=False, points=200, widths=[0.9])
plt.violinplot(nl_strongI, positions=[3], showextrema=False, points=200, widths=[0.9])
plt.violinplot(nl_strongE, positions=[4], showextrema=False, points=200, widths=[0.9])
plt.violinplot(nl_allEI, positions=[6], showextrema=False, points=200, widths=[0.9])
plt.violinplot(nl_unused, positions=[7], showextrema=False, points=200, widths=[0.9])
plt.plot([0.75,1.25],[np.median(nl_weakI),np.median(nl_weakI)],'k-')
plt.plot([1.75,2.25],[np.median(nl_weakE),np.median(nl_weakE)],'k-')
plt.plot([2.75,3.25],[np.median(nl_strongI),np.median(nl_strongI)],'k-')
plt.plot([3.75,4.25],[np.median(nl_strongE),np.median(nl_strongE)],'k-')
plt.plot([5.75,6.25],[np.median(nl_allEI),np.median(nl_allEI)],'k-')
plt.plot([6.75,7.25],[np.median(nl_unused),np.median(nl_unused)],'k-')
plt.ylim(-1.5,1.5)
plt.ylabel('Pre-Post R')
# stats comparing groups
statistic, weakp = stats.mannwhitneyu(nl_weakI, y=nl_weakE)
statistic, strongp = stats.mannwhitneyu(nl_strongI, y=nl_strongE)
statistic, Ip = stats.mannwhitneyu(nl_strongI, y=nl_weakI)
statistic, Ep = stats.mannwhitneyu(nl_strongE, y=nl_weakE)
statistic, unusedp = stats.mannwhitneyu(nl_unused, y=nl_allEI)
plt.text(1.25,1.25, 'p = ' + str(np.round(weakp,3)))
plt.text(3.25,1.25, 'p = ' + str(np.round(strongp,3)))
plt.text(6.25,1.25, 'p = ' + str(np.round(unusedp,3)))
plt.text(1.5,-1.2, 'p = ' + str(np.round(Ip,3)))
plt.text(3,-1.2, 'p = ' + str(np.round(Ep,3)))
# group indicators
plt.text(1-0.25,-1.4, 'weakI')
plt.text(2-0.25,-1.4, 'weakE')
plt.text(3-0.25,-1.4, 'strongI')
plt.text(4-0.25,-1.4, 'strongE')
plt.text(6-0.25,-1.4, 'used')
plt.text(7-0.25,-1.4, 'unused')

# flicker plot props
plt.subplot(3,4,5)
plt.plot(np.arange(binwidths/2,100+binwidths/2,binwidths), np.nanmedian(binned_flick_Rs, axis=1), '-', linewidth=2, color=[1,0.3,0.3])
plt.ylabel('Similarity to Post-Syn. Flicker Response (R)')
plt.xlabel('Cumulative Input Fraction')
plt.title('Cnx-Weighted Sum of Pre-Syn. Flicker Responses')
plt.xlim(0,100)
plt.ylim(-1.1,1.1)
plt.subplot(3,4,6)
plt.plot([20,20],[-1,1],':',color=[0.5,0.5,0.5])
plt.scatter(first_input_infrac, flick_first_input_corrs, s=20, color=[0.2,0.2,0.2])
# correlation coefficient
corrmat = np.corrcoef(np.asarray([first_input_infrac,flick_first_input_corrs]))
fn_cn_R = np.round(corrmat[0,1],2)
plt.text(50,-0.7,'R = ' + str(fn_cn_R))
# shuffle with replacement
fn_var = flick_first_input_corrs
cn_var = first_input_infrac
# remove nans
nl_cn_var = cn_var[~np.isnan(fn_var)]
nl_fn_var = fn_var[~np.isnan(fn_var)]
shuffs_fn_var = np.random.choice(nl_fn_var, size=(len(nl_fn_var),10000), replace=True)
# calculate correlation coefficient distribution
shuffled_corrs = np.corrcoef((np.append(nl_cn_var.reshape(-1,1), shuffs_fn_var, axis=1)).T)[0,1:]
# plot correlation coefficients
plt.subplot(3,4,7)
plt.violinplot(shuffled_corrs, positions=[4], showextrema=False, points=200, widths=[0.9])
plt.plot([3.75,4.25],[np.quantile(shuffled_corrs,0.025),np.quantile(shuffled_corrs,0.025)],'k:',linewidth=0.5)
plt.plot([3.75,4.25],[np.quantile(shuffled_corrs,0.975),np.quantile(shuffled_corrs,0.975)],'k:',linewidth=0.5)
x = plt.scatter(4, fn_cn_R, color=[0.2,0.2,0.2], s=20)
# set ticks
x.axes.get_xaxis().set_ticks([4])
x.axes.get_xaxis().set_ticklabels(['Corr. Coef. Null']);
# test for significance in a two-sided test
if np.logical_or(fn_cn_R > np.quantile(shuffled_corrs,0.975), fn_cn_R < np.quantile(shuffled_corrs,0.025)):
    plt.plot(4,0,'k*')

# plot binned means
plt.subplot(3,4,6)
pk_bins = np.digitize(first_input_infrac, bins=np.arange(15/2,75+15/2,15))
pk_corr_binned_means = np.zeros(len(np.arange(15/2,75+15/2,15)))
for bin in range(0,len(np.arange(15/2,75+15/2,15))):
    pk_corr_binned_means[bin] = np.median(flick_first_input_corrs[pk_bins==bin])
# manually fill in empty bin with mean of prior and latter
pk_corr_binned_means[3] = (pk_corr_binned_means[2]+pk_corr_binned_means[4])/2
plt.plot(np.arange(15/2,75,15), pk_corr_binned_means, '-', linewidth=2, color=[1,0.3,0.3])
plt.xlabel('Input Fraction of Largest Input (%)')
plt.title('Correlation vs. Input Fraction for Largest Input')
plt.xlim(0,80)
plt.ylim(-1.1,1.1)
plt.subplot(3,4,7)
for ind in range(0,len(infrac_types)):
    plt.plot([1,2,3],[flick_first_input_corrs[ind],flick_last_input_corrs[ind],flick_nobig_input_corrs[ind]],'.-',linewidth=0.5,color=[0.5,0.5,0.5])
plt.plot([1,2,3],[np.median(flick_first_input_corrs),np.median(flick_last_input_corrs),np.median(flick_nobig_input_corrs)],'.-', linewidth=2,color=[1,0.3,0.3])
# signed-rank test against null that distributions are the same
statistic, pvalue = stats.wilcoxon(flick_first_input_corrs, y=flick_last_input_corrs)
plt.text(1.15, 1.02, 'p = ' + str(np.round(pvalue,3)))
statistic, pvalue = stats.wilcoxon(flick_last_input_corrs, y=flick_nobig_input_corrs)
plt.text(2.15, 1.02, 'p = ' + str(np.round(pvalue,3)))
statistic, pvalue = stats.wilcoxon(flick_first_input_corrs, y=flick_nobig_input_corrs)
plt.text(1.65, -1.08, 'p = ' + str(np.round(pvalue,3)))
plt.title('Largest Input vs. All Inputs vs. All-Largest')
plt.xlim(0.5,4.5)
plt.ylim(-1.1,1.1)

plt.subplot(3,4,8)
# define plot groups (weak vs. strong, E vs. I), multiply I groups by -1 to "sign correct" correlations
weakE = flick_vec[np.logical_and(infrac_vec < infrac_thresh, infrac_vec > 0)]
strongE = thresh_flick_vec[thresh_infrac_vec > 0]
weakI = flick_vec[np.logical_and(infrac_vec > -1*infrac_thresh, infrac_vec < 0)]
strongI = thresh_flick_vec[thresh_infrac_vec < 0]
allEI = flick_vec[np.nonzero(infrac_vec)]
# remove nans from each plot group
nl_weakE = weakE[~np.isnan(weakE)]
nl_strongE = strongE[~np.isnan(strongE)]
nl_weakI = weakI[~np.isnan(weakI)]
nl_strongI = strongI[~np.isnan(strongI)]
nl_allEI = allEI[~np.isnan(allEI)]
nl_unused = flick_unused_vec[~np.isnan(flick_unused_vec)]
# plot violins of each distribution
plt.violinplot(nl_weakI, positions=[1], showextrema=False, points=200, widths=[0.9])
plt.violinplot(nl_weakE, positions=[2], showextrema=False, points=200, widths=[0.9])
plt.violinplot(nl_strongI, positions=[3], showextrema=False, points=200, widths=[0.9])
plt.violinplot(nl_strongE, positions=[4], showextrema=False, points=200, widths=[0.9])
plt.violinplot(nl_allEI, positions=[6], showextrema=False, points=200, widths=[0.9])
plt.violinplot(nl_unused, positions=[7], showextrema=False, points=200, widths=[0.9])
plt.plot([0.75,1.25],[np.median(nl_weakI),np.median(nl_weakI)],'k-')
plt.plot([1.75,2.25],[np.median(nl_weakE),np.median(nl_weakE)],'k-')
plt.plot([2.75,3.25],[np.median(nl_strongI),np.median(nl_strongI)],'k-')
plt.plot([3.75,4.25],[np.median(nl_strongE),np.median(nl_strongE)],'k-')
plt.plot([5.75,6.25],[np.median(nl_allEI),np.median(nl_allEI)],'k-')
plt.plot([6.75,7.25],[np.median(nl_unused),np.median(nl_unused)],'k-')
plt.ylim(-1.5,1.5)
plt.ylabel('Pre-Post R')
# stats comparing groups
statistic, weakp = stats.mannwhitneyu(nl_weakI, y=nl_weakE)
statistic, strongp = stats.mannwhitneyu(nl_strongI, y=nl_strongE)
statistic, Ip = stats.mannwhitneyu(nl_strongI, y=nl_weakI)
statistic, Ep = stats.mannwhitneyu(nl_strongE, y=nl_weakE)
statistic, unusedp = stats.mannwhitneyu(nl_unused, y=nl_allEI)
plt.text(1.25,1.25, 'p = ' + str(np.round(weakp,3)))
plt.text(3.25,1.25, 'p = ' + str(np.round(strongp,3)))
plt.text(6,1.25, 'p = ' + str(np.round(unusedp,4)))
plt.text(1.5,-1.2, 'p = ' + str(np.round(Ip,3)))
plt.text(3,-1.2, 'p = ' + str(np.round(Ep,5)))
# group indicators
plt.text(1-0.25,-1.4, 'weakI')
plt.text(2-0.25,-1.4, 'weakE')
plt.text(3-0.25,-1.4, 'strongI')
plt.text(4-0.25,-1.4, 'strongE')
plt.text(6-0.25,-1.4, 'used')
plt.text(7-0.25,-1.4, 'unused')


# save it
fname = '/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all cell summaries/temporal similarity vs cumulative input fraction.pdf'
tempcumfig.savefig(fname, format='pdf', orientation='landscape')

#%% cumulative input fraction statistics

np.median(total_infracs)
np.mean(total_infracs)
np.std(total_infracs)
np.min(total_infracs)
np.max(total_infracs)
np.quantile(total_infracs, 0.75)
np.quantile(total_infracs, 0.25)

infracdistfig = plt.figure(figsize=(5,4))
inf_hist = np.histogram(total_infracs,bins=11,range=(-5,105))
plt.plot(np.arange(0,110,10),inf_hist[0])
plt.ylabel('Number of cell types')
plt.xlabel('%'' of total input by main dataset cell types')

# save it
fname = '/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all cell summaries/total input fraction distribution.pdf'
infracdistfig.savefig(fname, format='pdf', orientation='landscape')

#%% plot input fraction distribution and relevant thresholds

infdistfig = plt.figure(figsize=(8,5))
infrac_hist = np.histogram(np.abs(infrac_data)[np.nonzero(infrac_data)], bins=np.logspace(-0.3,1.7,num=40))
binnormhist = infrac_hist[0]/np.diff(infrac_hist[1])
plt.semilogx(infrac_hist[1][:len(infrac_hist[1])-1],binnormhist/np.max(binnormhist))
plt.plot([2,2], [0,1], 'k:')
plt.plot([5,5], [0,1], 'k:')
plt.plot([10,10], [0,1], 'k:')
plt.ylabel('Relative probability')
plt.xlabel('Input fraction (%)')

# save it
fname = '/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all cell summaries/infrac_distribution_main_dataset.pdf'
infdistfig.savefig(fname, format='pdf', orientation='landscape')


#%% build data tables for full-data similarity analyses

# container arrays for cross-type similarity
cTRF_corrs_table = np.zeros((len(unique_types),len(unique_types)))
sTRF_corrs_table = np.zeros((len(unique_types),len(unique_types)))
flick_corrs_table = np.zeros((len(unique_types),len(unique_types)))
pc_dists_table = np.zeros((len(unique_types),len(unique_types)))
Nbytype = np.zeros(len(unique_types))

# populate tables one pair at a time
for ind1 in range(0,len(unique_types)):
    first_label = unique_types[ind1]
    # define N for cell type
    Nbytype[ind1] = len(scraped_log[scraped_log[:,14] == first_label,1])
    cTRF1 = mean_concat_TRFs[:,unique_types == first_label]
    sTRF1 = mean_concat_surr_TRFs[:,unique_types == first_label]
    flick1 = mean_concat_flicks[:,unique_types == first_label]
    pc1 = mean_pc_coords[:,unique_types == first_label]
    for ind2 in range(ind1+1,len(unique_types)):
        second_label = unique_types[ind2]
        cTRF2 = mean_concat_TRFs[:,unique_types == second_label]
        sTRF2 = mean_concat_surr_TRFs[:,unique_types == second_label]
        flick2 = mean_concat_flicks[:,unique_types == second_label]
        pc2 = mean_pc_coords[:,unique_types == second_label]
        # build corrcoef input arrays
        cTRFs = np.append(cTRF1.reshape(-1,1), cTRF2.reshape(-1,1), axis=1)
        sTRFs = np.append(sTRF1.reshape(-1,1), sTRF2.reshape(-1,1), axis=1)
        flicks_array = np.append(flick1.reshape(-1,1), flick2.reshape(-1,1), axis=1)
        # perform correlations
        cTRF_corrs_table[ind1,ind2] = np.corrcoef(cTRFs, rowvar=False)[0,1]
        sTRF_corrs_table[ind1,ind2] = np.corrcoef(sTRFs, rowvar=False)[0,1]
        flick_corrs_table[ind1,ind2] = np.corrcoef(flicks_array, rowvar=False)[0,1]
        # calculate PC space distance
        pc_dists_table[ind1,ind2] = np.linalg.norm(pc1-pc2)

# reflect across the main diagonal to make table lookup easier in the weighting steps
lower_inds = np.nonzero(np.tril(np.ones((len(unique_types),len(unique_types)))))
cTRF_corrs_table[lower_inds] = cTRF_corrs_table.T[lower_inds]
sTRF_corrs_table[lower_inds] = sTRF_corrs_table.T[lower_inds]
flick_corrs_table[lower_inds] = flick_corrs_table.T[lower_inds]
pc_dists_table[lower_inds] = pc_dists_table.T[lower_inds]

# fill diagonals with 1s (correlations) or 0s (distances)
np.fill_diagonal(cTRF_corrs_table, 1)
np.fill_diagonal(sTRF_corrs_table, 1)
np.fill_diagonal(flick_corrs_table, 1)
np.fill_diagonal(pc_dists_table, 0)

#%% clustering on full dataset and examining how weak vs strong inputs sample functional subtypes
from sklearn.cluster import AffinityPropagation

infrac_thresh=10
inclustfig = plt.figure(figsize=(20, 20))

# CLUSTER BLOCK
# define matrix to cluster on
input_matrix = pc_dists_table
input_labels = unique_types
# remove cell types giving NaN distances
input_labels = input_labels[~np.isnan(input_matrix)[0]]
input_matrix = input_matrix[~np.isnan(input_matrix)[0],:][:,~np.isnan(input_matrix)[0]]
# train and fit the agglomerative clustering model
clust_model = AffinityPropagation(max_iter=500, convergence_iter=25, random_state=0)
clust_IDs = clust_model.fit_predict(input_matrix)
# reorganize the distances matrix by cluster. within each cluster, sort cell types by mean distance, from closest to furthest
resorted_data = np.zeros(input_matrix.shape[0]).reshape(1,-1)
resorted_types = np.full(1,'',dtype='<U32')
for clust in range(0,np.max(clust_IDs)+1):
    clust_sortinds = np.argsort(np.mean(np.abs(input_matrix[clust_IDs==clust,:]),axis=1))
    resorted_data = np.append(resorted_data, input_matrix[clust_IDs==clust,:][clust_sortinds,:], axis=0)
    resorted_types = np.append(resorted_types, input_labels[clust_IDs==clust][clust_sortinds])
# trim initialization axis
resorted_data = resorted_data[1:,:]
resorted_types = resorted_types[1:]
# repeat to sort matrix columns
reresorted_data = np.zeros(resorted_data.shape[0]).reshape(-1,1)
for clust in range(0,np.max(clust_IDs)+1):
    clust_sortinds = np.argsort(np.mean(np.abs(resorted_data[:,clust_IDs==clust]),axis=0))
    reresorted_data = np.append(reresorted_data, resorted_data[:,clust_IDs==clust][:,clust_sortinds], axis=1)
reresorted_data = reresorted_data[:,1:]
# define number of functional clusters
clustered_types = resorted_types
num_fn_clusts = np.max(clust_IDs)+1
fn_clust_IDs = clust_IDs
# plotting
plt.title('PC space distance, nclust =' + str(num_fn_clusts))
x = plt.imshow(reresorted_data, cmap='pink', clim=[10,80])
plt.colorbar(shrink=0.5,aspect=8,ticks=[10,45,80])
# set ticks
x.axes.get_yaxis().set_ticks(np.arange(0,len(resorted_types),1))
x.axes.get_yaxis().set_ticklabels(resorted_types);
x.axes.get_xaxis().set_ticks([])
input_labels[clust_IDs==0]
fname = '/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all cell summaries/all cell types functional clustering.pdf'
inclustfig.savefig(fname, format='pdf', orientation='landscape')

#%% initialize containers for binned cluster IDs and n_clusters per cell type
strongbins = np.zeros(np.max(clust_IDs)+1)
weakbins = np.zeros(np.max(clust_IDs)+1)
n_strong = np.zeros(len(infrac_types))
n_weak = np.zeros(len(infrac_types))
n_in_strong = np.zeros(len(infrac_types))
n_in_weak = np.zeros(len(infrac_types))

# container for weak/strong occupancy vectors by cell type
occ_mat = np.zeros((len(infrac_types),2*num_fn_clusts))

# based on clust_IDs, ask for each infrac_type which clusters inputs come from, above or below infrac_thresh
for ind in range(0,len(infrac_types)):
    post_label = infrac_types[ind]
    weak_inds = np.where(np.logical_and(np.abs(infrac_data[:,ind])<infrac_thresh, np.abs(infrac_data[:,ind])>0))[0]
    weak_input_types = infrac_types[weak_inds]
    strong_inds = np.where(np.abs(infrac_data[:,ind])>=infrac_thresh)[0]
    strong_input_types = infrac_types[strong_inds]
    # containers for cluster identities
    strongclusts = np.full(len(strong_input_types),np.nan)
    weakclusts = np.full(len(weak_input_types),np.nan)
    # bin into strong/weak vectors according to cluster identity of each input by looping over inputs and adding 1 to the corresponding cluster bin
    for input in range(0,len(strong_input_types)):
        input_clust = clust_IDs[resorted_types == strong_input_types[input]]
        occ_mat[ind,0+input_clust] = occ_mat[ind,0+input_clust] + 1
        strongbins[input_clust] = strongbins[input_clust]+1
        strongclusts[input] = input_clust
    for input in range(0,len(weak_input_types)):
        input_clust = clust_IDs[resorted_types == weak_input_types[input]]
        occ_mat[ind,num_fn_clusts+input_clust] = occ_mat[ind,num_fn_clusts+input_clust] + 1
        weakbins[input_clust] = weakbins[input_clust]+1
        weakclusts[input] = input_clust
    # count number of unique clusters represented in strong or weak by type
    n_weak[ind] = len(np.unique(weakclusts))
    n_strong[ind] = len(np.unique(strongclusts))
    # counter number of cell types represented by strong or weak
    n_in_weak[ind] = len(weak_input_types)
    n_in_strong[ind] = len(strong_input_types)

plt.subplot(4,4,4)
plt.bar(np.arange(0,num_fn_clusts,1),strongbins/np.sum(strongbins)-weakbins/np.sum(weakbins),color=[0.3,0.3,0.3])
plt.plot([-0.5,num_fn_clusts-0.5],[0,0],'k:')
plt.ylim(-0.3,0.3)
plt.ylabel('Sampling Probability Difference')
plt.xlabel('Cluster Identity')
plt.subplot(4,4,8)
str_hist = np.histogram(n_strong,bins=num_fn_clusts,range=(0.5,num_fn_clusts+0.5))
str_pdf = str_hist[0]/np.sum(str_hist[0])
str_cdf = np.cumsum(str_pdf)
plt.plot(np.append(0,str_hist[1][1:]-0.5),np.append(0,str_cdf),color=[0.3,0.8,0.3])
plt.ylabel('Cumulative Probability')
plt.xlabel('Num Clusters Represented, Strong Inputs')
plt.subplot(4,4,12)
wk_hist = np.histogram(n_weak,bins=num_fn_clusts,range=(0.5,num_fn_clusts+0.5))
wk_pdf = wk_hist[0]/np.sum(wk_hist[0])
wk_cdf = np.cumsum(wk_pdf)
plt.plot(np.append(0,wk_hist[1][1:]-0.5),np.append(0,wk_cdf),color=[0.8,0.3,0.8])
plt.ylabel('Cumulative Probability')
plt.xlabel('Num Clusters Represented, Weak Inputs')


# CLUSTER BLOCK
# define matrix to cluster on
input_matrix = occ_mat
input_labels = infrac_types
# train and fit the agglomerative clustering model
clust_model = AffinityPropagation(max_iter=500, convergence_iter=25, random_state=0)
clust_IDs = clust_model.fit_predict(input_matrix)
# reorganize the distances matrix by cluster. within each cluster, sort cell types by mean distance, from closest to furthest
resorted_data = np.zeros(input_matrix.shape[1]).reshape(1,-1)
resorted_types = np.full(1,'',dtype='<U32')
for clust in range(0,np.max(clust_IDs)+1):
    clust_sortinds = np.argsort(np.mean(np.abs(input_matrix[clust_IDs==clust,:]),axis=1))
    resorted_data = np.append(resorted_data, input_matrix[clust_IDs==clust,:][clust_sortinds,:], axis=0)
    resorted_types = np.append(resorted_types, input_labels[clust_IDs==clust][clust_sortinds])
# trim initialization axis
resorted_data = resorted_data[1:,:]
resorted_types = resorted_types[1:]

plt.subplot(2,4,5)
plt.title('Functional Cluster Occupancy')
x = plt.imshow(resorted_data, cmap='binary')
plt.plot([num_fn_clusts-0.5,num_fn_clusts-0.5],[-0.5,len(infrac_types)-0.5],'r-')
plt.colorbar(shrink=0.5,aspect=8)
# set ticks
x.axes.get_yaxis().set_ticks(np.arange(0,len(resorted_types),1))
x.axes.get_yaxis().set_ticklabels(resorted_types);
x.axes.get_xaxis().set_ticks([4.5,14.5])
x.axes.get_xaxis().set_ticklabels(['strong','weak']);

plt.subplot(2,4,6)
plt.title('Joint Occupancy')
joint_occ_mat = resorted_data[:,0:num_fn_clusts]*resorted_data[:,num_fn_clusts:]
x = plt.imshow(joint_occ_mat, cmap='binary',clim=[0,6])
# set ticks
x.axes.get_yaxis().set_ticks(np.arange(0,len(resorted_types),1))
x.axes.get_yaxis().set_ticklabels(resorted_types);
x.axes.get_xaxis().set_ticks([0,2,4,6,8])
x.axes.get_xaxis().set_ticklabels([0,2,4,6,8]);
plt.xlabel('Functional Cluster')

plt.subplot(4,4,16)
str_hist = np.histogram(n_in_strong,bins=15,range=(0,30))
str_norm = str_hist[0]/np.sum(str_hist[0])
plt.plot(str_hist[1][1:],str_norm,color=[0.3,0.8,0.3])
wk_hist = np.histogram(n_in_weak,bins=15,range=(0,30))
wk_norm = wk_hist[0]/np.sum(wk_hist[0])
plt.plot(wk_hist[1][1:],wk_norm,color=[0.8,0.3,0.8])
plt.ylabel('Probability')
plt.xlabel('Raw Number of Inputs')


# NULL GENERATION
# iterate over fn_clust_IDs to pull out cluster sampling of high N cell types
infrac_type_clusts = []
for ind in range(0,len(fn_clust_IDs)):
    current_label = clustered_types[ind]
    if np.any(infrac_types == current_label):
        infrac_type_clusts = np.append(infrac_type_clusts, fn_clust_IDs[ind])

plt.subplot(4,4,7)
# bin the fn_clust_IDs vector to get the number of cells types in each cluster
ID_hist = np.histogram(fn_clust_IDs,bins=num_fn_clusts,range=(-0.5,num_fn_clusts-0.5))
ID_norm = ID_hist[0]/np.sum(ID_hist[0])
# plot number of cell types per cluster
plt.bar(np.arange(0,num_fn_clusts,1),ID_hist[0],color=[0.7,0.7,0.7],align='edge',width=-0.4)
# bin the infrac_type_clusts vector to get the random probability of each cluster getting drawn
ID_hist = np.histogram(infrac_type_clusts,bins=num_fn_clusts,range=(-0.5,num_fn_clusts-0.5))
ID_norm = ID_hist[0]/np.sum(ID_hist[0])
# plot number of cell types per cluster
plt.bar(np.arange(0,num_fn_clusts,1),ID_hist[0],color=[0.3,0.3,0.3],align='edge',width=0.4)
plt.ylabel('Number of Cell Types')
plt.xlabel('Cluster Identity')

# bin the number of cell types from each class (strong, weak) to know how many samples to draw for each simulated neuron in the null set
str_hist = np.histogram(n_in_strong,bins=30,range=(0,30))
str_norm = str_hist[0]/np.sum(str_hist[0])
wk_hist = np.histogram(n_in_weak,bins=30,range=(0,30))
wk_norm = wk_hist[0]/np.sum(wk_hist[0])
# first, pull a set of random n_samples to draw from the raw # of inputs distributions
n_runs = 10000
n_samples = np.random.choice(np.round(str_hist[1][1:]), n_runs, p=str_norm).astype('int')
# container array for num_clusters from each random draw
nclusts_null = np.zeros(n_runs)
# then, for each n_samples, simulate random pulls from clusters, then find the number of unique clusters drawn
for run in range(0,n_runs):
    samples = np.random.choice(np.arange(0,num_fn_clusts,1), n_samples[run], p=ID_norm)
    nclusts_null[run] = len(np.unique(samples))
# plot the null distribution from this random run
null_hist = np.histogram(nclusts_null,bins=num_fn_clusts,range=(0.5,num_fn_clusts+0.5))
null_norm = null_hist[0]/np.sum(null_hist[0])
null_cdf = np.cumsum(null_norm)
plt.subplot(4,4,8)
plt.plot(np.append(0,null_hist[1][1:]-0.5),np.append(0,null_cdf),color=[0,0.4,0])
# compare to empirical distribution with a k-s test
kr_str = stats.kstest(nclusts_null,n_strong)
plt.text(6,0.1,'p = ' + str(np.round(kr_str.pvalue,4)))

# repeat for weak input null
n_samples = np.random.choice(np.round(wk_hist[1][1:]), n_runs, p=wk_norm).astype('int')
# container array for num_clusters from each random draw
nclusts_null = np.zeros(n_runs)
# then, for each n_samples, simulate random pulls from clusters, then find the number of unique clusters drawn
for run in range(0,n_runs):
    samples = np.random.choice(np.arange(0,num_fn_clusts,1), n_samples[run], p=ID_norm)
    nclusts_null[run] = len(np.unique(samples))
# plot the null distribution from this random run
null_hist = np.histogram(nclusts_null,bins=num_fn_clusts,range=(0.5,num_fn_clusts+0.5))
null_norm = null_hist[0]/np.sum(null_hist[0])
null_cdf = np.cumsum(null_norm)
plt.subplot(4,4,12)
plt.plot(np.append(0,null_hist[1][1:]-0.5),np.append(0,null_cdf),color=[0.4,0,0.4])
# compare to empirical distribution with a k-s test
kr_wk = stats.kstest(nclusts_null,n_weak)
plt.text(0,0.8,'p = ' + str(np.round(kr_wk.pvalue,4)))

# save it
plt.suptitle('Sampling of Functional Clusters Among Strong or Weak Inputs, infrac_thresh=' + str(infrac_thresh))
fname = '/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all cell summaries/sampling of functional clusters for weak-strong inputs(infrac_thresh=' + str(infrac_thresh) + ').pdf'
inclustfig.savefig(fname, format='pdf', orientation='portrait')
