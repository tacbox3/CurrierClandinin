# This script compares chromatic selectivity to R7/8 input fractions across cell types.

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

# load R7/R8 table pulled from Nern website
book = xlrd.open_workbook('/Users/tcurrier/Desktop/Clandinin Lab/Connectomics/Nern2024_IPR_InFracs_noPR.xls')
sheet = book.sheet_by_name('Sheet1')
data = [[sheet.cell_value(r, c) for c in range(sheet.ncols)] for r in range(sheet.nrows)]
infrac_data = np.asarray(data)

# fill in empty cells with 0s, cut out cell type and metric labels, then convert to float
infrac_data[infrac_data==''] = '0.0'
ipr_target_types = infrac_data[0,1:]
infrac_metric_labels = infrac_data[1:,0]
infrac_data = infrac_data[1:,1:].astype('float')

#%% create data vectors with metrics corresponding to each cell type in the data

# pull list of unique cell type labels in my data
utype_indices = np.unique(scraped_log[:,14]) != ''
unique_types = np.unique(scraped_log[:,14])[utype_indices]
Nbytype = np.zeros(len(unique_types))

# initialize container vectors for each connectome spectral variable
R7p_frac = np.zeros(len(unique_types))
R7y_frac = np.zeros(len(unique_types))
R8p_frac = np.zeros(len(unique_types))
R8y_frac = np.zeros(len(unique_types))
R7_frac = np.zeros(len(unique_types))
R8_frac = np.zeros(len(unique_types))
p_frac = np.zeros(len(unique_types))
y_frac = np.zeros(len(unique_types))
total_ipr_frac = np.zeros(len(unique_types))

# loop over cell types and populate vectors
for ind in range(0,len(unique_types)):
    # define cell type to summarize
    label = unique_types[ind]
    # define N on first loop through cell types
    current_cells = scraped_log[scraped_log[:,14] == label,1]
    current_n = len(current_cells)
    Nbytype[ind] = current_n
    # grab data for current cell type and add it to metric-specific vector
    if np.any(ipr_target_types == label):
        R7p_frac[ind] = infrac_data[np.where(infrac_metric_labels == 'R7p input fraction')[0][0], np.where(ipr_target_types==label)[0][0]]
        R7y_frac[ind] = infrac_data[np.where(infrac_metric_labels == 'R7y input fraction')[0][0], np.where(ipr_target_types==label)[0][0]]
        R8p_frac[ind] = infrac_data[np.where(infrac_metric_labels == 'R8p input fraction')[0][0], np.where(ipr_target_types==label)[0][0]]
        R8y_frac[ind] = infrac_data[np.where(infrac_metric_labels == 'R8y input fraction')[0][0], np.where(ipr_target_types==label)[0][0]]
        R7_frac[ind] = infrac_data[np.where(infrac_metric_labels == 'R7 input fraction')[0][0], np.where(ipr_target_types==label)[0][0]]
        R8_frac[ind] = infrac_data[np.where(infrac_metric_labels == 'R8 input fraction')[0][0], np.where(ipr_target_types==label)[0][0]]
        p_frac[ind] = infrac_data[np.where(infrac_metric_labels == 'p input fraction')[0][0], np.where(ipr_target_types==label)[0][0]]
        y_frac[ind] = infrac_data[np.where(infrac_metric_labels == 'y input fraction')[0][0], np.where(ipr_target_types==label)[0][0]]
        total_ipr_frac[ind] = infrac_data[np.where(infrac_metric_labels == 'total IPR input fraction')[0][0], np.where(ipr_target_types==label)[0][0]]

# load pre-calculated physiology metrics
uvpi_c = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all_spectral_pref_index_CENTER.npy')
uvpi_s = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all_spectral_pref_index_SURROUND.npy')
pc_weights = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all_PC_weights.npy')
pc4 = pc_weights[:,4]
pc5 = pc_weights[:,4]

# pre-compute summary metrics - joint absolute preference index ranges from 0 (no color spectivity) to 2 (perfect selectivity / opponency in both center and surround)
uvpi_joint = np.abs(uvpi_c)+np.abs(uvpi_s)
spectral_pc_joint = np.abs(pc4)+np.abs(pc5)

# initialize physiology container vectors
uvpi_c_mean = np.zeros(len(unique_types))
uvpi_s_mean = np.zeros(len(unique_types))
uvpi_joint_mean = np.zeros(len(unique_types))
pc4_mean = np.zeros(len(unique_types))
pc5_mean = np.zeros(len(unique_types))
spectral_pc_joint_mean = np.zeros(len(unique_types))

for ind in range(0,len(unique_types)):
    # define cell type to summarize
    current_label = unique_types[ind]
    current_cells = scraped_log[scraped_log[:,14] == current_label,1]
    current_n = len(current_cells)
    # UVPI
    label_uvpi_c = uvpi_c[scraped_log[:,14] == current_label]
    uvpi_c_mean[ind] = np.nanmedian(label_uvpi_c)
    label_uvpi_s = uvpi_s[scraped_log[:,14] == current_label]
    uvpi_s_mean[ind] = np.nanmedian(label_uvpi_s)
    label_uvpi_joint = uvpi_joint[scraped_log[:,14] == current_label]
    uvpi_joint_mean[ind] = np.nanmedian(label_uvpi_joint)
    # spectrally selective PCs
    label_pc4 = pc4[scraped_log[:,14] == current_label]
    pc4_mean[ind] = np.nanmedian(label_pc4)
    label_pc5 = pc5[scraped_log[:,14] == current_label]
    pc5_mean[ind] = np.nanmedian(label_pc5)
    label_spectral_pc_joint = spectral_pc_joint[scraped_log[:,14] == current_label]
    spectral_pc_joint_mean[ind] = np.nanmedian(label_spectral_pc_joint)

#%% trim off cell types with no IPR input
ipr_inds = np.unique(np.append(np.nonzero(R7_frac),np.nonzero(R8_frac)))

# trim metadata
thresh_types = unique_types[ipr_inds]
thresh_Nbytype = Nbytype[ipr_inds]

# trim connectome vectors
thresh_R7p_frac = R7p_frac[ipr_inds]
thresh_R7y_frac = R7y_frac[ipr_inds]
thresh_R8p_frac = R8p_frac[ipr_inds]
thresh_R8y_frac = R8y_frac[ipr_inds]
thresh_R7_frac = R7_frac[ipr_inds]
thresh_R8_frac = R8_frac[ipr_inds]
thresh_y_frac = y_frac[ipr_inds]
thresh_p_frac = p_frac[ipr_inds]
thresh_total_ipr_frac = total_ipr_frac[ipr_inds]

# trim physiology vectors
thresh_uvpi_c_mean = uvpi_c_mean[ipr_inds]
thresh_uvpi_s_mean = uvpi_s_mean[ipr_inds]
thresh_uvpi_joint_mean = uvpi_joint_mean[ipr_inds]
thresh_pc4_mean = pc4_mean[ipr_inds]
thresh_pc5_mean = pc5_mean[ipr_inds]
thresh_spectral_pc_joint_mean = spectral_pc_joint_mean[ipr_inds]

#%% set threshold N and number of shuffles for null distribution
thresh_n = 0
numshuffles = 10000

# trim off thresholded cell types in connectome vectors
thresh_R7p_frac = thresh_R7p_frac[thresh_Nbytype>thresh_n]
thresh_R7y_frac = thresh_R7y_frac[thresh_Nbytype>thresh_n]
thresh_R8p_frac = thresh_R8p_frac[thresh_Nbytype>thresh_n]
thresh_R8y_frac = thresh_R8y_frac[thresh_Nbytype>thresh_n]
thresh_R7_frac = thresh_R7_frac[thresh_Nbytype>thresh_n]
thresh_R8_frac = thresh_R8_frac[thresh_Nbytype>thresh_n]
thresh_y_frac = thresh_y_frac[thresh_Nbytype>thresh_n]
thresh_p_frac = thresh_p_frac[thresh_Nbytype>thresh_n]
thresh_total_ipr_frac = thresh_total_ipr_frac[thresh_Nbytype>thresh_n]

# trim off thresholded cell types in physiology vectors
thresh_uvpi_c_mean = thresh_uvpi_c_mean[thresh_Nbytype>thresh_n]
thresh_uvpi_s_mean = thresh_uvpi_s_mean[thresh_Nbytype>thresh_n]
thresh_uvpi_joint_mean = thresh_uvpi_joint_mean[thresh_Nbytype>thresh_n]
thresh_pc4_mean = thresh_pc4_mean[thresh_Nbytype>thresh_n]
thresh_pc5_mean = thresh_pc5_mean[thresh_Nbytype>thresh_n]
thresh_spectral_pc_joint_mean = thresh_spectral_pc_joint_mean[thresh_Nbytype>thresh_n]

# trim metadata (this needs to be done last here because I'm overwriting the thresh_Nbytype variable)
thresh_types = thresh_types[thresh_Nbytype>thresh_n]
thresh_Nbytype = thresh_Nbytype[thresh_Nbytype>thresh_n]

#%% plot full UVPI distribution, plus direct IPR input correlations
uvpifig = plt.figure(figsize=(15, 10))

# FULL CENTER UVPI DISTRIBUTION
var_to_plot = uvpi_c_mean
full_var_vec = uvpi_c
plt.subplot(2,1,1)
# sort by means
sorted_inds = np.argsort(var_to_plot)
sorted_types = unique_types[sorted_inds]
# plot a box for each cell type meeting threshold
for n in range(0,len(sorted_types)):
    current_label = sorted_types[n]
    label_v = full_var_vec[scraped_log[:,14] == current_label]
    # remove NaNs
    label_v = label_v[~np.isnan(label_v)]
    plt.scatter(np.full(len(label_v),n),label_v,color=[0,0,0],s=8,alpha=0.3)
    if np.any(ipr_target_types == current_label):
        plt.text(n, 1.3, sorted_types[n], rotation='vertical', color=[0.2,0.5,0.5])
        plt.text(n, 1.15, str(len(label_v)), color=[0.2,0.5,0.5])
    else:
        plt.text(n, 1.3, sorted_types[n], rotation='vertical')
        plt.text(n, 1.15, str(len(label_v)))
# plot means and plot params
plt.scatter(np.arange(0,len(unique_types),1), var_to_plot[sorted_inds], color=[0,0,0], s=30)
plt.plot([-1,len(var_to_plot)+1],[0,0],'k:')
plt.xlim([-1,len(var_to_plot)+1])
plt.ylabel('Center Spectral Preference Index')

# CENTER UVPI VS. R7 FRACTION
plt.subplot(2,3,4)
fn_var = thresh_uvpi_c_mean
cn_var = thresh_R7_frac
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
plt.plot(np.arange(-2,22,1),np.arange(-2,22,1)*fn_cn_k+popt[1],'r-')
plt.title('R = ' + str(fn_cn_R) + ', k = ' + str(np.round(fn_cn_k,5)) + ' /%')
plt.ylabel('Center SPI')
plt.xlabel('R7 Input (''%'' of total)')
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
plt.subplot(2,6,11)
plt.title('Corr. Coef. Nulls')
plt.violinplot(shuffled_corrs, positions=[1], showextrema=False, points=200, widths=[0.9])
plt.plot([0.75,1.25],[np.quantile(shuffled_corrs,0.025),np.quantile(shuffled_corrs,0.025)],'k:',linewidth=0.5)
plt.plot([0.75,1.25],[np.quantile(shuffled_corrs,0.975),np.quantile(shuffled_corrs,0.975)],'k:',linewidth=0.5)
plt.scatter(1, fn_cn_R, color=[0.2,0.2,0.2], s=20)
# test for significance in a two-sided test
if np.logical_or(fn_cn_R > np.quantile(shuffled_corrs,0.975), fn_cn_R < np.quantile(shuffled_corrs,0.025)):
    plt.plot(1,0,'k*')
# plot slopes
plt.subplot(2,6,12)
plt.title('Slope Nulls')
plt.violinplot(shuffled_slopes, positions=[1], showextrema=False, points=200, widths=[0.9])
plt.plot([0.75,1.25],[np.quantile(shuffled_slopes,0.025),np.quantile(shuffled_slopes,0.025)],'k:',linewidth=0.5)
plt.plot([0.75,1.25],[np.quantile(shuffled_slopes,0.975),np.quantile(shuffled_slopes,0.975)],'k:',linewidth=0.5)
plt.scatter(1, fn_cn_k, color=[0.2,0.2,0.2], s=20)
# test for significance in a two-sided test
if np.logical_or(fn_cn_k > np.quantile(shuffled_slopes,0.975), fn_cn_k < np.quantile(shuffled_slopes,0.025)):
    plt.plot(1,0,'k*')

# JOINT ABSOLUTE UVPI VS. TOTAL IPR FRACTION
plt.subplot(2,3,5)
fn_var = thresh_uvpi_joint_mean
cn_var = thresh_total_ipr_frac
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
plt.plot(np.arange(-3,45,1),np.arange(-3,45,1)*fn_cn_k+popt[1],'r-')
plt.title('R = ' + str(fn_cn_R) + ', k = ' + str(np.round(fn_cn_k,5)) + ' /%')
plt.ylabel('Center + Surround Joint Absolute SPI')
plt.xlabel('Total IPR Input (''%'' of total)')
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
plt.subplot(2,6,11)
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
plt.subplot(2,6,12)
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
fname = '/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all cell summaries/spectral selectivity connectome correlations.pdf'
uvpifig.savefig(fname, format='pdf', orientation='landscape')
