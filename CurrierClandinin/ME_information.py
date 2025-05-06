# mutual information analysis between PC-space physiological distance and varying connectomic distances. There are a number of one-time-only blocks that must be run before the final plotting (generates interim .npy summary files). Navigate to the bottom of the script to find these blocks.
# import packages
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import cv2
import csv
import xlrd
import warnings
from pathlib import Path
from datetime import datetime
from scipy import ndimage
from scipy import stats
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from scipy.optimize import curve_fit

def line(x, k, b):
    y = k*x + b
    return (y)

# define function for calculating MI
def mutual_information(img1, img2, bins=100):
    """
    Measure the mutual information of the given two images. Author: Jin Jeon

    Parameters
    ----------
    img1: nii image data read via nibabel
    img2: nii image data read via nibabel
    bins: optional (default=100)
        bin size of the histogram

    Returns
    -------
    calculated mutual information: float
    """
    hist_2d, x_edges, y_edges = np.histogram2d(img1.ravel(), img2.ravel(), bins)

    # convert bins counts to probability values
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1)  # marginal x over y
    py = np.sum(pxy, axis=0)  # marginal y over x
    px_py = px[:, None] * py[None, :]  # broadcast to multiply marginals

    # now we can do the calculation using the pxy, px_py 2D arrays
    nonzeros = pxy > 0  # filer out the zero values
    return np.sum(pxy[nonzeros] * np.log(pxy[nonzeros] / px_py[nonzeros]))

def MI_null_dist(input_mat, comparison_mat, bins=100, numshuffles=10000, replacement=True):
    """
    Calculate a null set of MI values via random resampling of input_mat. Author: TAC

    Parameters
    ----------
    input_mat: 2-D array to shuffle
    comparison_mat: 2-D array, MI is calculated between comparison_mat and shuffled versions of input_mat
    bins: optional int (default=100)
        bin size of the histogram
    numshuffles: optional int (default=10000)
        number of shuffles to execute
    replacement: optional bool (default=True)
        whether or not to replace drawn values during resampling

    Returns
    -------
    calculated mutual information values: 1-D array of length numshuffles
    """
    # create a set of numshuffles matrices that draw randomly from the connectivity matrices, excluding diagonal values (identity, distance = 0)
    shuffled_mats = np.random.choice(input_mat[np.nonzero(np.tril(input_mat))], size=(input_mat.shape+(numshuffles,)), replace=replacement)

    # identity matrix for indexing and a container vector for MI null distribution
    full_mat = np.ones(input_mat.shape)
    shuff_MIs = np.zeros(numshuffles)

    # main MI calculation loop
    for n in range(0,numshuffles):
        # define current matrix
        current_matrix = shuffled_mats[:,:,n]
        # mirror the current shuffled matrix over the main diagonal
        current_matrix = np.where(np.tril(full_mat)==1, current_matrix, current_matrix.T)
        # manually overwrite identity to the main diagonal of each shuffled matrix
        current_matrix[np.diag_indices_from(current_matrix)] = 0
        # calculate the MI between the current matrix and the physiology distance matrix, add to vector
        shuff_MIs[n] = mutual_information(current_matrix, comparison_mat, bins=bins)

    return shuff_MIs

# disable runtime and deprecation warnings - dangerous!
warnings.filterwarnings("ignore")

#%% load and preprocess physiology data
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

# pull list of unique cell type labels in my data
utype_indices = np.unique(scraped_log[:,14]) != ''
unique_types = np.unique(scraped_log[:,14])[utype_indices]

# load physiology data as PC loadings ('full' = 100 PCs for all cells, 'all' = first 6 PCs for all cells)
pc_weights = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/full_PC_weights.npy')
# pc_weights = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all_PC_weights.npy')

# container arrays for N and mean PC coordinates
Nbytype = np.zeros(len(unique_types))
mean_pc_coords = np.zeros((pc_weights.shape[1],len(unique_types)))

# loop over cell types and populate physiology data arrays
for ind in range(0,len(unique_types)):
    # define cell type to summarize
    current_label = unique_types[ind]
    # define N for cell type
    current_cells = scraped_log[scraped_log[:,14] == current_label,1]
    current_n = len(np.unique(current_cells))
    Nbytype[ind] = current_n
    # grab pc data for current cell type and find the mean coordinate
    label_pc_coords = pc_weights[scraped_log[:,14] == current_label,:]
    mean_pc_coords[:,ind] = np.nanmean(label_pc_coords, axis=0)

# container array for cross-type PC space distances
pc_dists_table = np.zeros((len(unique_types),len(unique_types)))

# populate table one pair at a time
for ind1 in range(0,len(unique_types)):
    first_label = unique_types[ind1]
    pc1 = mean_pc_coords[:,unique_types == first_label]
    for ind2 in range(ind1+1,len(unique_types)):
        second_label = unique_types[ind2]
        pc2 = mean_pc_coords[:,unique_types == second_label]
        # calculate PC space distance
        pc_dists_table[ind1,ind2] = np.linalg.norm(pc1-pc2)

# reflect across the main diagonal to make table lookup easier in the weighting steps
lower_inds = np.nonzero(np.tril(np.ones((len(unique_types),len(unique_types)))))
pc_dists_table[lower_inds] = pc_dists_table.T[lower_inds]

# fill diagonals with 0s (identity distance)
np.fill_diagonal(pc_dists_table, 0)

# remove NaN distances (incomplete data for N=1 cell types Tm5a and Dm12)
phys_dist_types = unique_types[~np.isnan(pc_dists_table[0,:])]
phys_Nbytype = Nbytype[~np.isnan(pc_dists_table[0,:])]
pc_dists_table = pc_dists_table[:,~np.isnan(pc_dists_table[0,:])][~np.isnan(pc_dists_table[0,:]),:]


#%% open Reiser data - must run the ONE-TIME ONLY blocks at the bottom of the script before proceeding further!

# Reiser type-to-type connectivity matrix as input fractions
Rsr_conn_table = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Connectomics/NernReiser2024 Data/__EXTRACTED__type2type_input_fractions.npy')
Rsr_conn_table_counts = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Connectomics/NernReiser2024 Data/__EXTRACTED__type2type_counts.npy')
Rsr_types = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Connectomics/NernReiser2024 Data/__EXTRACTED__cell_types.npy')

# Reiser type-to-type weighted Jaccard distances (pre-computed)
Rsr_allcnx_wJdist_mat = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Connectomics/NernReiser2024 Data/Rsr_weighted_distances_culled.npy')
Rsr_incnx_wJdist_mat = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Connectomics/NernReiser2024 Data/Rsr_weighted_distances_inputs_only_culled.npy')
Rsr_outcnx_wJdist_mat = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Connectomics/NernReiser2024 Data/Rsr_weighted_distances_outputs_only_culled.npy')

# transpose the counts table and divide each column by the sum of all counts for that column, converts raw counts into OUTPUT fractions!
Rsr_type2type = Rsr_conn_table_counts.T
Rsr_type2type_outfrac = np.zeros(Rsr_type2type.shape)
for pre_type in range(0,Rsr_type2type.shape[1]):
    current_inputs = Rsr_type2type[:,pre_type]
    Rsr_type2type_outfrac[:,pre_type] = current_inputs/np.sum(current_inputs)
# turn nans to 0 (required for cell types that provide no output)
Rsr_type2type_outfrac[np.isnan(Rsr_type2type_outfrac)] = 0

# add the transposed outfrac table as another set of ~750 rows, such that each column now contains a list of all input connections, followed by a list of all output connections. this can be used for "all connections" vs. "inputs only" predictability comparison
Rsr_inout_conn_table = np.append(Rsr_conn_table, Rsr_type2type_outfrac, axis=0)

# retain only columns (post-synaptic types) that are present in phys_dist_types. keep all rows.
Rsr_conn_subset = np.zeros((Rsr_conn_table.shape[0],len(phys_dist_types)))
Rsr_counts_subset = np.zeros((Rsr_conn_table_counts.shape[0],len(phys_dist_types)))
Rsr_inout_conn_subset = np.zeros((Rsr_inout_conn_table.shape[0],len(phys_dist_types)))
Rsr_out_conn_subset = np.zeros((Rsr_type2type_outfrac.shape[0],len(phys_dist_types)))
for n in range(0,len(phys_dist_types)):
    current_type = phys_dist_types[n]
    if current_type == 'T4':
        current_type = 'T4a'
    Rsr_conn_subset[:,n] = Rsr_conn_table[:,Rsr_types==current_type].reshape(-1)
    Rsr_counts_subset[:,n] = Rsr_conn_table_counts[:,Rsr_types==current_type].reshape(-1)
    Rsr_inout_conn_subset[:,n] = Rsr_inout_conn_table[:,Rsr_types==current_type].reshape(-1)
    Rsr_out_conn_subset[:,n] = Rsr_type2type_outfrac[:,Rsr_types==current_type].reshape(-1)


#%% process Reiser data

# define N threshold and range of input fraction thresholds for testing MI change, as well as bandwidths to test
thresh_n = 2
thresh_infracs = np.linspace(0,20,40)/100
thresh_counts = np.linspace(0,20000,40)
bandwidth1 = 0.1 #units of input fraction - values from 0.3-0.5 maximize, 0.65+ recapitulate full threshold
bandwidth2 = 0.2
bandwidth3 = 0.3

# copy phys table, type list and type Ns to retain syntax from earlier section
culled_Nbytype = phys_Nbytype
culled_pc_dists_table = pc_dists_table
culled_phys_dist_types = phys_dist_types

# create a thresholded copy of the postsynaptic cell type list
thresh_phys_dist_types = culled_phys_dist_types[culled_Nbytype > thresh_n]

# calculate "all connections" MI and null distribution by shuffle, along with "output only"
# container array for cross-type connection distances
Rsr_allcnx_dist_mat = np.zeros((len(phys_dist_types),len(phys_dist_types)))
Rsr_outcnx_dist_mat = np.zeros((len(phys_dist_types),len(phys_dist_types)))

# calculate type-to-type distances based on inputs vectors
for ind1 in range(0,len(phys_dist_types)):
    v1 = Rsr_inout_conn_subset[:,ind1]
    w1 = Rsr_out_conn_subset[:,ind1]
    for ind2 in range(ind1+1,len(phys_dist_types)):
        v2 = Rsr_inout_conn_subset[:,ind2]
        w2 = Rsr_out_conn_subset[:,ind2]
        Rsr_allcnx_dist_mat[ind1,ind2] = np.linalg.norm(v1-v2)
        Rsr_outcnx_dist_mat[ind1,ind2] = np.linalg.norm(w1-w2)

# reflect across the main diagonal to make table lookup easier in the weighting steps
lower_inds = np.nonzero(np.tril(np.ones((len(phys_dist_types),len(phys_dist_types)))))
Rsr_allcnx_dist_mat[lower_inds] = Rsr_allcnx_dist_mat.T[lower_inds]
Rsr_outcnx_dist_mat[lower_inds] = Rsr_outcnx_dist_mat.T[lower_inds]

# fill diagonals with 0s (identity distance)
np.fill_diagonal(Rsr_allcnx_dist_mat, 0)
np.fill_diagonal(Rsr_outcnx_dist_mat, 0)

# create a thresholded copy of the phys and conn distance tables (rows and columns removed for cell types where N does not meet threshold)
thresh_pc_dists_table = culled_pc_dists_table[culled_Nbytype > thresh_n,:][:,culled_Nbytype > thresh_n]
thresh_Rsr_allcnx_dist_mat = Rsr_allcnx_dist_mat[culled_Nbytype > thresh_n,:][:,culled_Nbytype > thresh_n]
thresh_Rsr_outcnx_dist_mat = Rsr_outcnx_dist_mat[culled_Nbytype > thresh_n,:][:,culled_Nbytype > thresh_n]

# calculate MI between thresholded distance matrices
allcnx_mi = mutual_information(thresh_Rsr_allcnx_dist_mat, thresh_pc_dists_table)
outcnx_mi = mutual_information(thresh_Rsr_outcnx_dist_mat, thresh_pc_dists_table)

# find MI null distribution
allcnx_mi_null = MI_null_dist(thresh_Rsr_allcnx_dist_mat, thresh_pc_dists_table)
outcnx_mi_null = MI_null_dist(thresh_Rsr_outcnx_dist_mat, thresh_pc_dists_table)

# initialize vectors for MI by infrac threshold
mi = np.zeros(len(thresh_infracs))
mi_below = np.zeros(len(thresh_infracs))
wJ_mi = np.zeros(len(thresh_infracs))
wJ_mi_below = np.zeros(len(thresh_infracs))
mi_bw1 = np.zeros(len(thresh_infracs))
mi_bw2 = np.zeros(len(thresh_infracs))
mi_bw3 = np.zeros(len(thresh_infracs))

for thr in range(0,len(thresh_infracs)):
    # threshold the Reiser connection table subset @ current_thresh
    current_thresh = thresh_infracs[thr]
    count_thresh = thresh_counts[thr]
    Rsr_conn_subset_thresh = np.where(Rsr_conn_subset > current_thresh, Rsr_conn_subset, 0)
    Rsr_conn_subset_bw1 = np.where(np.logical_and(Rsr_conn_subset > current_thresh, Rsr_conn_subset < current_thresh + bandwidth1), Rsr_conn_subset, 0)
    Rsr_conn_subset_bw2 = np.where(np.logical_and(Rsr_conn_subset > current_thresh, Rsr_conn_subset < current_thresh + bandwidth2), Rsr_conn_subset, 0)
    Rsr_conn_subset_bw3 = np.where(np.logical_and(Rsr_conn_subset > current_thresh, Rsr_conn_subset < current_thresh + bandwidth3), Rsr_conn_subset, 0)
    Rsr_counts_subset_thresh = np.where(Rsr_counts_subset > count_thresh, Rsr_counts_subset, 0)
    # container array for cross-type connection distances
    Rsr_incnx_dist_mat = np.zeros((len(phys_dist_types),len(phys_dist_types)))
    Rsr_bw1_dist_mat = np.zeros((len(phys_dist_types),len(phys_dist_types)))
    Rsr_bw2_dist_mat = np.zeros((len(phys_dist_types),len(phys_dist_types)))
    Rsr_bw3_dist_mat = np.zeros((len(phys_dist_types),len(phys_dist_types)))
    Rsr_wJ_dist_mat = np.zeros((len(phys_dist_types),len(phys_dist_types)))
    # calculate type-to-type distances based on inputs vectors
    for ind1 in range(0,len(phys_dist_types)):
        v1 = Rsr_conn_subset_thresh[:,ind1]
        w1 = Rsr_counts_subset_thresh[:,ind1]
        x1 = Rsr_conn_subset_bw1[:,ind1]
        y1 = Rsr_conn_subset_bw2[:,ind1]
        z1 = Rsr_conn_subset_bw3[:,ind1]
        for ind2 in range(ind1+1,len(phys_dist_types)):
            v2 = Rsr_conn_subset_thresh[:,ind2]
            w2 = Rsr_counts_subset_thresh[:,ind2]
            x2 = Rsr_conn_subset_bw1[:,ind2]
            y2 = Rsr_conn_subset_bw2[:,ind2]
            z2 = Rsr_conn_subset_bw3[:,ind2]
            # calculate Euclidean distance between the two cell types as points in a high-dim conn space
            Rsr_incnx_dist_mat[ind1,ind2] = np.linalg.norm(v1-v2)
            Rsr_wJ_dist_mat[ind1,ind2] = np.linalg.norm(w1-w2)
            Rsr_bw1_dist_mat[ind1,ind2] = np.linalg.norm(x1-x2)
            Rsr_bw2_dist_mat[ind1,ind2] = np.linalg.norm(y1-y2)
            Rsr_bw3_dist_mat[ind1,ind2] = np.linalg.norm(z1-z2)
            # # weighted Jaccard distance is 1 - (sum of all smallest values between w1 and w2 divided by the sum of all largest values between v1 and v2)
            # Rsr_wJ_dist_mat[ind1,ind2] = 1 - (np.sum(np.nanmin(np.append(w1,w2,axis=1),axis=1)) / np.sum(np.nanmax(np.append(w1,w2,axis=1),axis=1)))

    # reflect across the main diagonal to make table lookup easier in the weighting steps
    Rsr_incnx_dist_mat[lower_inds] = Rsr_incnx_dist_mat.T[lower_inds]
    Rsr_wJ_dist_mat[lower_inds] = Rsr_wJ_dist_mat.T[lower_inds]
    Rsr_bw1_dist_mat[lower_inds] = Rsr_bw1_dist_mat.T[lower_inds]
    Rsr_bw2_dist_mat[lower_inds] = Rsr_bw2_dist_mat.T[lower_inds]
    Rsr_bw3_dist_mat[lower_inds] = Rsr_bw3_dist_mat.T[lower_inds]
    # fill diagonals with 0s (identity distance)
    np.fill_diagonal(Rsr_incnx_dist_mat, 0)
    np.fill_diagonal(Rsr_wJ_dist_mat, 0)
    np.fill_diagonal(Rsr_bw1_dist_mat, 0)
    np.fill_diagonal(Rsr_bw2_dist_mat, 0)
    np.fill_diagonal(Rsr_bw3_dist_mat, 0)
    # create a thresholded copy of the phys and conn distance tables (rows and columns removed for cell types where N does not meet threshold)
    thresh_pc_dists_table = culled_pc_dists_table[culled_Nbytype > thresh_n,:][:,culled_Nbytype > thresh_n]
    thresh_Rsr_incnx_dist_mat = Rsr_incnx_dist_mat[culled_Nbytype > thresh_n,:][:,culled_Nbytype > thresh_n]
    thresh_Rsr_wJ_dist_mat = Rsr_wJ_dist_mat[culled_Nbytype > thresh_n,:][:,culled_Nbytype > thresh_n]
    thresh_Rsr_bw1_dist_mat = Rsr_bw1_dist_mat[culled_Nbytype > thresh_n,:][:,culled_Nbytype > thresh_n]
    thresh_Rsr_bw2_dist_mat = Rsr_bw2_dist_mat[culled_Nbytype > thresh_n,:][:,culled_Nbytype > thresh_n]
    thresh_Rsr_bw3_dist_mat = Rsr_bw3_dist_mat[culled_Nbytype > thresh_n,:][:,culled_Nbytype > thresh_n]

    # calculate MI between thresholded distance matrices
    mi[thr] = mutual_information(thresh_Rsr_incnx_dist_mat, thresh_pc_dists_table)
    mi_bw1[thr] = mutual_information(thresh_Rsr_bw1_dist_mat, thresh_pc_dists_table)
    mi_bw2[thr] = mutual_information(thresh_Rsr_bw2_dist_mat, thresh_pc_dists_table)
    mi_bw3[thr] = mutual_information(thresh_Rsr_bw3_dist_mat, thresh_pc_dists_table)
    wJ_mi[thr] = mutual_information(thresh_Rsr_wJ_dist_mat, thresh_pc_dists_table)

    # save copy of optimal case for image plotting below
    if thr == 17:
        thresh_optimal_bw_dist_mat = thresh_Rsr_bw3_dist_mat

    # find MI null distribution only when threshold is 0 (no connections cut)
    if current_thresh == 0:
        incnx_mi_null = MI_null_dist(thresh_Rsr_incnx_dist_mat, thresh_pc_dists_table)
        incounts_mi_null = MI_null_dist(thresh_Rsr_wJ_dist_mat, thresh_pc_dists_table)

    # REPEAT PROCESS FOR CONNECTIONS --BELOW-- THRESHOLD!
    if current_thresh > 0:
        Rsr_conn_subset_thresh = np.where(Rsr_conn_subset > current_thresh, 0, Rsr_conn_subset)
        Rsr_counts_subset_thresh = np.where(Rsr_counts_subset > count_thresh, 0, Rsr_counts_subset)
        Rsr_incnx_dist_mat = np.zeros((len(phys_dist_types),len(phys_dist_types)))
        Rsr_wJ_dist_mat = np.zeros((len(phys_dist_types),len(phys_dist_types)))
        # calculate type-to-type distances based on inputs vectors
        for ind1 in range(0,len(phys_dist_types)):
            v1 = Rsr_conn_subset_thresh[:,ind1]
            w1 = Rsr_counts_subset_thresh[:,ind1]
            for ind2 in range(ind1+1,len(phys_dist_types)):
                v2 = Rsr_conn_subset_thresh[:,ind2]
                w2 = Rsr_counts_subset_thresh[:,ind2]
                Rsr_incnx_dist_mat[ind1,ind2] = np.linalg.norm(v1-v2)
                Rsr_wJ_dist_mat[ind1,ind2] = np.linalg.norm(w1-w2)
                # Rsr_wJ_dist_mat[ind1,ind2] = 1 - (np.sum(np.min(np.append(w1,w2,axis=1),axis=1)) / np.sum(np.max(np.append(w1,w2,axis=1),axis=1)))
        Rsr_incnx_dist_mat[lower_inds] = Rsr_incnx_dist_mat.T[lower_inds]
        Rsr_wJ_dist_mat[lower_inds] = Rsr_wJ_dist_mat.T[lower_inds]
        np.fill_diagonal(Rsr_incnx_dist_mat, 0)
        np.fill_diagonal(Rsr_wJ_dist_mat, 0)
        thresh_pc_dists_table = culled_pc_dists_table[culled_Nbytype > thresh_n,:][:,culled_Nbytype > thresh_n]
        thresh_Rsr_incnx_dist_mat = Rsr_incnx_dist_mat[culled_Nbytype > thresh_n,:][:,culled_Nbytype > thresh_n]
        thresh_Rsr_wJ_dist_mat = Rsr_wJ_dist_mat[culled_Nbytype > thresh_n,:][:,culled_Nbytype > thresh_n]
        # calculate MI between thresholded distance matrices
        mi_below[thr] = mutual_information(thresh_Rsr_incnx_dist_mat, thresh_pc_dists_table)
        wJ_mi_below[thr] = mutual_information(thresh_Rsr_wJ_dist_mat, thresh_pc_dists_table)

# threshold and process weighted Jaccard distance matrices
thresh_Rsr_allcnx_wJdist_mat = Rsr_allcnx_wJdist_mat[culled_Nbytype > thresh_n,:][:,culled_Nbytype > thresh_n]
allcnx_wJ_mi = mutual_information(thresh_Rsr_allcnx_wJdist_mat, thresh_pc_dists_table)
allcnx_wJ_mi_null = MI_null_dist(thresh_Rsr_allcnx_wJdist_mat, thresh_pc_dists_table)
thresh_Rsr_incnx_wJdist_mat = Rsr_incnx_wJdist_mat[culled_Nbytype > thresh_n,:][:,culled_Nbytype > thresh_n]
incnx_wJ_mi = mutual_information(thresh_Rsr_incnx_wJdist_mat, thresh_pc_dists_table)
incnx_wJ_mi_null = MI_null_dist(thresh_Rsr_incnx_wJdist_mat, thresh_pc_dists_table)
thresh_Rsr_outcnx_wJdist_mat = Rsr_outcnx_wJdist_mat[culled_Nbytype > thresh_n,:][:,culled_Nbytype > thresh_n]
outcnx_wJ_mi = mutual_information(thresh_Rsr_outcnx_wJdist_mat, thresh_pc_dists_table)
outcnx_wJ_mi_null = MI_null_dist(thresh_Rsr_outcnx_wJdist_mat, thresh_pc_dists_table)


#%% top X connection analysis

# define N threshold and range of top x connections
thresh_n = 2
top_x = np.linspace(1,30,30).astype('int')

# initialize vectors for MI by top x connections
mi_topx = np.zeros(len(top_x))
mi_botx = np.zeros(len(top_x))

# build the rank list for each column in the connection table subset
conn_ranklist = np.argsort(Rsr_conn_subset,axis=0)

# calculate MI for conn distances computed on the top or bottom x conncections of each cell type
for thr in range(0,len(top_x)):
    # define the current "x" (number of top or bottom connections to retain for distance computation)
    current_x = top_x[thr]
    # container array for cross-type connection distances
    Rsr_topx_dist_mat = np.zeros((len(phys_dist_types),len(phys_dist_types)))
    Rsr_botx_dist_mat = np.zeros((len(phys_dist_types),len(phys_dist_types)))
    # calculate type-to-type distances based on top/bottom x input vectors
    for ind1 in range(0,len(phys_dist_types)):
        # define current cell input vector
        cell1 = Rsr_conn_subset[:,ind1]
        # identify the first non-zero-returning index of the ranklist for the given column (cell type), use this index to define the "starting point" of the bottom-x list
        bot_rank = np.min(np.nonzero(cell1[conn_ranklist[:,ind1]]))
        # define threshold value of connection weight that represents top x connections
        top_thr = cell1[conn_ranklist[-(current_x),ind1]]
        # define threshold value of connection weight that represents bottom x connections
        bot_thr = cell1[conn_ranklist[bot_rank+(current_x-1),ind1]]
        # threshold the cell vector and save for distance computation below
        top1 = np.where(cell1 >= top_thr, cell1, 0)
        bot1 = np.where(cell1 <= bot_thr, cell1, 0)
        # repeat computations for second cell type
        for ind2 in range(ind1+1,len(phys_dist_types)):
            cell2 = Rsr_conn_subset[:,ind2]
            bot_rank = np.min(np.nonzero(cell2[conn_ranklist[:,ind2]]))
            top_thr = cell2[conn_ranklist[-(current_x),ind2]]
            bot_thr = cell2[conn_ranklist[bot_rank+(current_x-1),ind2]]
            top2 = np.where(cell2 >= top_thr, cell2, 0)
            bot2 = np.where(cell2 <= bot_thr, cell2, 0)
            # calculate Euclidean distance between the two cell types as points in a high-dim conn space
            Rsr_topx_dist_mat[ind1,ind2] = np.linalg.norm(top1-top2)
            Rsr_botx_dist_mat[ind1,ind2] = np.linalg.norm(bot1-bot2)
    # reflect across the main diagonal to make table lookup easier in the weighting steps
    Rsr_topx_dist_mat[lower_inds] = Rsr_topx_dist_mat.T[lower_inds]
    Rsr_botx_dist_mat[lower_inds] = Rsr_botx_dist_mat.T[lower_inds]
    # fill diagonals with 0s (identity distance)
    np.fill_diagonal(Rsr_topx_dist_mat, 0)
    np.fill_diagonal(Rsr_botx_dist_mat, 0)
    # create a thresholded copy of the conn distance tables (rows and columns removed for cell types where N does not meet threshold)
    thresh_Rsr_topx_dist_mat = Rsr_topx_dist_mat[culled_Nbytype > thresh_n,:][:,culled_Nbytype > thresh_n]
    thresh_Rsr_botx_dist_mat = Rsr_botx_dist_mat[culled_Nbytype > thresh_n,:][:,culled_Nbytype > thresh_n]
    # calculate MI between thresholded distance matrices
    mi_topx[thr] = mutual_information(thresh_Rsr_topx_dist_mat, thresh_pc_dists_table)
    mi_botx[thr] = mutual_information(thresh_Rsr_botx_dist_mat, thresh_pc_dists_table)


#%% open FAFB data - this block must be run AFTER Reiser blocks, and Reiser blocks may not be run after this segment has completed - some variable names are reused and will create thresholding errors due to different numbers of eligible cell types in the two datasets

# FAFB all inputs/outputs Jaccard distance - un-recorded cell types manually removed
book = xlrd.open_workbook('/Users/tcurrier/Desktop/Clandinin Lab/Connectomics/FAFB Jaccard distances/FAFB_distances_culled.xls')
sheet = book.sheet_by_name('typetotypedistances')
data = [[sheet.cell_value(r, c) for c in range(sheet.ncols)] for r in range(sheet.nrows)]
FAFB_allcnx_dist = np.asarray(data)

# FAFB all inputs Jaccard distance - un-recorded cell types manually removed
book = xlrd.open_workbook('/Users/tcurrier/Desktop/Clandinin Lab/Connectomics/FAFB Jaccard distances/FAFB_distances_inputs_only_culled.xls')
sheet = book.sheet_by_name('FAFB_distances(inputs)')
data = [[sheet.cell_value(r, c) for c in range(sheet.ncols)] for r in range(sheet.nrows)]
FAFB_incnx_dist = np.asarray(data)

# grab cell labels and data as separate variables for FAFB tables (both use same format). Note that I've already manually renamed "conflict" cell types into Reiser nomenclature
FAFB_types = FAFB_allcnx_dist[0,1:]
FAFB_allcnx_dist_mat = FAFB_allcnx_dist[1:,1:].astype('float')
FAFB_incnx_dist_mat = FAFB_incnx_dist[1:,1:].astype('float')

# sort by cell type labels so connectome and physiology matricies use same indices per type
FAFB_allcnx_dist_mat = FAFB_allcnx_dist_mat[np.argsort(FAFB_types),:][:,np.argsort(FAFB_types)]
FAFB_incnx_dist_mat = FAFB_incnx_dist_mat[np.argsort(FAFB_types),:][:,np.argsort(FAFB_types)]
FAFB_types = np.sort(FAFB_types)

# remove entries corresponding to NaN-valued PC distances (see above section)
cull_labels = ['Dm12','Tm5a']
for label in cull_labels:
    FAFB_allcnx_dist_mat = FAFB_allcnx_dist_mat[FAFB_types != label,:][:,FAFB_types != label]
    FAFB_incnx_dist_mat = FAFB_incnx_dist_mat[FAFB_types != label,:][:,FAFB_types != label]
    FAFB_types = FAFB_types[FAFB_types != label]

#%% co-process FAFB and physiology matrices
# define threshold for plotting, N>2 ==> nclusts=5; N>0 ==> nclusts=10
# these nclusts were previously found in an unbiased manner via AffinityPropagation, but to keep the two sort direction comparison fair, we want to mandate a certain number of clusters for this final plot
thresh_n = 2
numshuffles = 10000

# cull cell types from the physiology table that were not included in the FAFB data - MeTu and PR types, plus Tm40, which is not a cell type in the FAFB dataset
cull_labels = ['R7p','R7y','R8p','R8y','MeTu1','MeTu4d','Tm40']
culled_Nbytype = phys_Nbytype
culled_pc_dists_table = pc_dists_table
culled_phys_dist_types = phys_dist_types
for label in cull_labels:
    culled_pc_dists_table = culled_pc_dists_table[culled_phys_dist_types != label,:][:,culled_phys_dist_types != label]
    culled_Nbytype = culled_Nbytype[culled_phys_dist_types != label]
    culled_phys_dist_types = culled_phys_dist_types[culled_phys_dist_types != label]

# create a thresholded copy of each table (rows and columns removed for cell types where N does not meet threshold)
thresh_pc_dists_table = culled_pc_dists_table[culled_Nbytype > thresh_n,:][:,culled_Nbytype > thresh_n]
thresh_FAFB_allcnx_dist_mat = FAFB_allcnx_dist_mat[culled_Nbytype > thresh_n,:][:,culled_Nbytype > thresh_n]
thresh_FAFB_incnx_dist_mat = FAFB_incnx_dist_mat[culled_Nbytype > thresh_n,:][:,culled_Nbytype > thresh_n]
# create a thresholded copy of each cell list (these should now match)
thresh_phys_dist_types = culled_phys_dist_types[culled_Nbytype > thresh_n]
thresh_FAFB_types = FAFB_types[culled_Nbytype > thresh_n]

# FAFB MUTUAL INFORMATION ANALYSIS - compute null distributions of MI for randomly resampled connectivity distances
allcnx_null = MI_null_dist(thresh_FAFB_allcnx_dist_mat, thresh_pc_dists_table)
incnx_null = MI_null_dist(thresh_FAFB_incnx_dist_mat, thresh_pc_dists_table)


#%% plot

MIfig = plt.figure(figsize=(18, 10))

# plot true MI value on top of null distributions
x = plt.subplot(231)
plt.violinplot(allcnx_mi_null, positions=[-3], showextrema=False, points=200, widths=[0.9])
plt.plot(-3, allcnx_mi, 'k.')
plt.plot([-3.25,-2.75], [np.quantile(allcnx_mi_null,0.025),np.quantile(allcnx_mi_null,0.025)],'k:',linewidth=1)
plt.plot([-3.25,-2.75], [np.quantile(allcnx_mi_null,0.975),np.quantile(allcnx_mi_null,0.975)],'k:',linewidth=1)
plt.violinplot(incnx_mi_null, positions=[-1], showextrema=False, points=200, widths=[0.9])
plt.plot(-1, mi[0], 'k.')
plt.plot([-1.25,-0.75], [np.quantile(incnx_mi_null,0.025),np.quantile(incnx_mi_null,0.025)],'k:',linewidth=1)
plt.plot([-1.25,-0.75], [np.quantile(incnx_mi_null,0.975),np.quantile(incnx_mi_null,0.975)],'k:',linewidth=1)
plt.ylabel('Mutual Information')
plt.title('Reiser Euclidean Connection Distances')
x.axes.get_xaxis().set_ticks([-3,-1])
x.axes.get_xaxis().set_ticklabels(['All Conn','Input Only']);

x = plt.subplot(232)
plt.violinplot(allcnx_wJ_mi_null, positions=[-2], showextrema=False, points=200, widths=[0.9])
plt.plot(-2, allcnx_wJ_mi, 'k.')
plt.plot([-2.25,-1.75], [np.quantile(allcnx_wJ_mi_null,0.025),np.quantile(allcnx_wJ_mi_null,0.025)],'k:',linewidth=1)
plt.plot([-2.25,-1.75], [np.quantile(allcnx_wJ_mi_null,0.975),np.quantile(allcnx_wJ_mi_null,0.975)],'k:',linewidth=1)
plt.violinplot(incnx_wJ_mi_null, positions=[-1], showextrema=False, points=200, widths=[0.9])
plt.plot(-1, incnx_wJ_mi, 'k.')
plt.plot([-1.25,-0.75], [np.quantile(incnx_wJ_mi_null,0.025),np.quantile(incnx_wJ_mi_null,0.025)],'k:',linewidth=1)
plt.plot([-1.25,-0.75], [np.quantile(incnx_wJ_mi_null,0.975),np.quantile(incnx_wJ_mi_null,0.975)],'k:',linewidth=1)

plt.violinplot(allcnx_null, positions=[0], showextrema=False, points=200, widths=[0.9])
plt.plot(0, mutual_information(thresh_FAFB_allcnx_dist_mat, thresh_pc_dists_table), 'k.')
plt.plot([-0.25,0.25],[np.quantile(allcnx_null,0.025),np.quantile(allcnx_null,0.025)],'k:',linewidth=0.5)
plt.plot([-0.25,0.25],[np.quantile(allcnx_null,0.975),np.quantile(allcnx_null,0.975)],'k:',linewidth=0.5)

plt.violinplot(incnx_null, positions=[1], showextrema=False, points=200, widths=[0.9])
plt.plot(1, mutual_information(thresh_FAFB_incnx_dist_mat, thresh_pc_dists_table), 'k.')
plt.plot([0.75,1.25],[np.quantile(incnx_null,0.025),np.quantile(incnx_null,0.025)],'k:',linewidth=0.5)
plt.plot([0.75,1.25],[np.quantile(incnx_null,0.975),np.quantile(incnx_null,0.975)],'k:',linewidth=0.5)

plt.ylabel('Mutual Information')
plt.title('Weighted Jaccard Connection Distances')
x.axes.get_xaxis().set_ticks([-2,-1,0,1])
x.axes.get_xaxis().set_ticklabels(['Rsr All','Rsr In','FAFB All','FAFB In']);



x = plt.subplot(233)
plt.violinplot(allcnx_wJ_mi_null/allcnx_wJ_mi, positions=[-2], showextrema=False, points=200, widths=[0.9])
plt.plot(-2, allcnx_wJ_mi/allcnx_wJ_mi, 'k.')
plt.plot([-2.25,-1.75], [np.quantile(allcnx_wJ_mi_null/allcnx_wJ_mi,0.025),np.quantile(allcnx_wJ_mi_null/allcnx_wJ_mi,0.025)],'k:',linewidth=1)
plt.plot([-2.25,-1.75], [np.quantile(allcnx_wJ_mi_null/allcnx_wJ_mi,0.975),np.quantile(allcnx_wJ_mi_null/allcnx_wJ_mi,0.975)],'k:',linewidth=1)

plt.violinplot(incnx_wJ_mi_null/allcnx_wJ_mi, positions=[-1], showextrema=False, points=200, widths=[0.9])
plt.plot(-1, incnx_wJ_mi/allcnx_wJ_mi, 'k.')
plt.plot([-1.25,-0.75], [np.quantile(incnx_wJ_mi_null/allcnx_wJ_mi,0.025),np.quantile(incnx_wJ_mi_null/allcnx_wJ_mi,0.025)],'k:',linewidth=1)
plt.plot([-1.25,-0.75], [np.quantile(incnx_wJ_mi_null/allcnx_wJ_mi,0.975),np.quantile(incnx_wJ_mi_null/allcnx_wJ_mi,0.975)],'k:',linewidth=1)

FAFB_all_mi = mutual_information(thresh_FAFB_allcnx_dist_mat, thresh_pc_dists_table)
plt.violinplot(allcnx_null/FAFB_all_mi, positions=[0], showextrema=False, points=200, widths=[0.9])
plt.plot(0, FAFB_all_mi/FAFB_all_mi, 'k.')
plt.plot([-0.25,0.25],[np.quantile(allcnx_null/FAFB_all_mi,0.025),np.quantile(allcnx_null/FAFB_all_mi,0.025)],'k:',linewidth=0.5)
plt.plot([-0.25,0.25],[np.quantile(allcnx_null/FAFB_all_mi,0.975),np.quantile(allcnx_null/FAFB_all_mi,0.975)],'k:',linewidth=0.5)

plt.violinplot(incnx_null/FAFB_all_mi, positions=[1], showextrema=False, points=200, widths=[0.9])
plt.plot(1, mutual_information(thresh_FAFB_incnx_dist_mat, thresh_pc_dists_table)/FAFB_all_mi, 'k.')
plt.plot([0.75,1.25],[np.quantile(incnx_null/FAFB_all_mi,0.025),np.quantile(incnx_null/FAFB_all_mi,0.025)],'k:',linewidth=0.5)
plt.plot([0.75,1.25],[np.quantile(incnx_null/FAFB_all_mi,0.975),np.quantile(incnx_null/FAFB_all_mi,0.975)],'k:',linewidth=0.5)

plt.ylabel('Relative Information')
plt.title('Weighted Jaccard Connection Distances')
x.axes.get_xaxis().set_ticks([-2,-1,0,1])
x.axes.get_xaxis().set_ticklabels(['Rsr All','Rsr In','FAFB All','FAFB In']);

# plot MI as a function of input fraction threshold
plt.subplot(234)
plt.plot([0,20], [np.quantile(incnx_mi_null,0.005),np.quantile(incnx_mi_null,0.005)],'k:',linewidth=1)
plt.plot([0,20], [np.quantile(incnx_mi_null,0.995),np.quantile(incnx_mi_null,0.995)],'k:',linewidth=1)
plt.plot(thresh_infracs*100, mi, color=[0.1,0.1,0.1], linewidth=1.5)
plt.plot(thresh_infracs[1:]*100, mi_below[1:], color=[0.1,0.1,0.8], linewidth=1.5)
plt.ylim([1.1, 1.8])
plt.title('MI Above/Below Input Fraction Threshold')
plt.xlabel('Input Fraction Threshold (%)')
plt.ylabel('Mutual Information')

plt.subplot(235)
plt.plot([0,20], [np.quantile(incnx_mi_null,0.005),np.quantile(incnx_mi_null,0.005)],'k:',linewidth=1)
plt.plot([0,20], [np.quantile(incnx_mi_null,0.995),np.quantile(incnx_mi_null,0.995)],'k:',linewidth=1)
plt.plot(thresh_infracs*100, mi, color=[0.1,0.1,0.1], linewidth=1.5)
plt.plot(thresh_infracs*100, mi_bw1, color=[0.8,0.7,0.1], linewidth=1.5)
plt.plot(thresh_infracs*100, mi_bw2, color=[0.8,0.4,0.1], linewidth=1.5)
plt.plot(thresh_infracs*100, mi_bw3, color=[0.8,0.1,0.1], linewidth=1.5)
plt.ylim([1.1, 1.8])
plt.title('MI by Input Fraction Band')
plt.xlabel('Lower Band Cutoff Input Fraction (%)')
plt.ylabel('Mutual Information')

plt.subplot(236)
plt.plot([0,30], [np.quantile(incnx_mi_null,0.005),np.quantile(incnx_mi_null,0.005)],'k:',linewidth=1)
plt.plot([0,30], [np.quantile(incnx_mi_null,0.995),np.quantile(incnx_mi_null,0.995)],'k:',linewidth=1)
plt.plot(top_x, mi_topx, color=[0.1,0.1,0.1], linewidth=1.5)
plt.plot(top_x[11:], mi_botx[11:], color=[0.1,0.1,0.8], linewidth=1.5)
plt.ylim([1.1, 1.8])
plt.title('MI for Top/Bottom X Inputs')
plt.ylabel('Mutual Information')
plt.xlabel('X Inputs')

# save it
fname = '/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all cell summaries/cnx-phys mutual information analysis.pdf'
MIfig.savefig(fname, format='pdf', orientation='landscape')

#%% difference between Jaccard distances computed on FAFB and Reiser data
diffig = plt.figure(figsize=(12,12))
plt.subplot(221)
plt.imshow(thresh_FAFB_allcnx_dist_mat-thresh_Rsr_allcnx_wJdist_mat, clim=(-0.25,0.25), cmap='PuOr'), plt.colorbar()
plt.title('All Cnx, FAFB-Rsr')
plt.subplot(222)
h,x = np.histogram(np.abs((thresh_FAFB_allcnx_dist_mat-thresh_Rsr_allcnx_wJdist_mat).flatten()),bins=100, range=(0,0.25))
plt.plot(x[1:],np.cumsum(h)/np.sum(h))
plt.ylabel('cumulative prob')
plt.xlabel('absolute deviation')
plt.subplot(223)
plt.imshow(thresh_FAFB_incnx_dist_mat-thresh_Rsr_incnx_wJdist_mat, clim=(-0.25,0.25), cmap='PuOr'), plt.colorbar()
plt.title('Input Only, FAFB-Rsr')
plt.subplot(224)
h,x = np.histogram(np.abs((thresh_FAFB_incnx_dist_mat-thresh_Rsr_incnx_wJdist_mat).flatten()),bins=100, range=(0,0.25))
plt.plot(x[1:],np.cumsum(h)/np.sum(h))
plt.ylabel('cumulative prob')
plt.xlabel('absolute deviation')

# save it
fname = '/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all cell summaries/Reiser-FAFB Jaccard Distance Differences.pdf'
diffig.savefig(fname, format='pdf', orientation='landscape')


#%% plot examples of sorted conn mirrored onto phys distances for all inputs case vs. optimal band case
similarityfig = plt.figure(figsize=(15, 15))
nclusts = 5

# PC DISTANCE VS. ALL INPUT CNX SIMILARITY
plt.subplot(3,3,3)
fn_var = np.tril(thresh_pc_dists_table).flatten()
cn_var = np.tril(thresh_Rsr_allcnx_dist_mat).flatten()
# remove nans
nl_cn_var = cn_var[~np.isnan(fn_var)]
nl_fn_var = fn_var[~np.isnan(fn_var)]
# remove self-similarity and zero-d out upper triangle
nl_fn_var = nl_fn_var[np.nonzero(nl_cn_var)]
nl_cn_var = nl_cn_var[np.nonzero(nl_cn_var)]
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
plt.scatter(nl_cn_var,nl_fn_var,color=[0,0,0],s=8)
plt.plot(np.arange(0,1,.01),np.arange(0,1,.01)*fn_cn_k+popt[1],'r-')
plt.text(0,70,'R = ' + str(fn_cn_R) + ', k = ' + str(np.round(fn_cn_k,2)))
plt.ylabel('PC Space Functional Distance')
plt.xlabel('Reiser All Inputs Distance')

# cluster on connection distance, plot physiology distance by sorted cnx clusts
# CLUSTER BLOCK
# define matrix to cluster on
input_matrix = thresh_Rsr_allcnx_dist_mat
input_labels = thresh_phys_dist_types
mirror_matrix = thresh_pc_dists_table
# train and fit the KMeans clustering model
clust_model = KMeans(n_clusters=nclusts, random_state=0)
clust_IDs = clust_model.fit_predict(input_matrix)
# reorganize the distances matrix by cluster. within each cluster, sort cell types by mean distance, from closest to furthest
resorted_data = np.zeros(input_matrix.shape[0]).reshape(1,-1)
resorted_mirror = np.zeros(mirror_matrix.shape[0]).reshape(1,-1)
resorted_types = np.full(1,'',dtype='<U32')
for clust in range(0,np.max(clust_IDs)+1):
    clust_sortinds = np.argsort(np.mean(np.abs(input_matrix[clust_IDs==clust,:]),axis=1))
    resorted_mirror = np.append(resorted_mirror, mirror_matrix[clust_IDs==clust,:][clust_sortinds,:], axis=0)
    resorted_data = np.append(resorted_data, input_matrix[clust_IDs==clust,:][clust_sortinds,:], axis=0)
    resorted_types = np.append(resorted_types, input_labels[clust_IDs==clust][clust_sortinds])
# trim initialization axis
resorted_mirror = resorted_mirror[1:,:]
resorted_data = resorted_data[1:,:]
resorted_types = resorted_types[1:]
# repeat to sort matrix columns
reresorted_data = np.zeros(resorted_data.shape[0]).reshape(-1,1)
reresorted_mirror = np.zeros(resorted_mirror.shape[0]).reshape(-1,1)
for clust in range(0,np.max(clust_IDs)+1):
    clust_sortinds = np.argsort(np.mean(np.abs(resorted_data[:,clust_IDs==clust]),axis=0))
    reresorted_mirror = np.append(reresorted_mirror, resorted_mirror[:,clust_IDs==clust][:,clust_sortinds], axis=1)
    reresorted_data = np.append(reresorted_data, resorted_data[:,clust_IDs==clust][:,clust_sortinds], axis=1)
reresorted_data = reresorted_data[:,1:]
reresorted_mirror = reresorted_mirror[:,1:]
# plotting
plt.subplot(3,3,1)
plt.title('Reiser All Input Dist, nclust =' + str(np.max(clust_IDs)+1))
x = plt.imshow(reresorted_data, cmap='cividis', clim=[0.2,0.8])
plt.colorbar(shrink=0.5,aspect=8,ticks=[0.2,0.8])
# set ticks
x.axes.get_yaxis().set_ticks([])
x.axes.get_xaxis().set_ticks([])
plt.subplot(3,3,2)
plt.title('Cluster-mirrored func distance')
x = plt.imshow(reresorted_mirror, cmap='pink', clim=[10,80])
plt.colorbar(shrink=0.5,aspect=8,ticks=[10,45,80])
# set ticks
x.axes.get_yaxis().set_ticks([])
x.axes.get_xaxis().set_ticks([])


# PC DISTANCE VS. BEST INPUT SUBSET CNX SIMILARITY
plt.subplot(3,3,6)
fn_var = np.tril(thresh_pc_dists_table).flatten()
cn_var = np.tril(thresh_optimal_bw_dist_mat).flatten()
# remove nans
nl_cn_var = cn_var[~np.isnan(fn_var)]
nl_fn_var = fn_var[~np.isnan(fn_var)]
# remove self-similarity and zero-d out upper triangle
nl_fn_var = nl_fn_var[np.nonzero(nl_cn_var)]
nl_cn_var = nl_cn_var[np.nonzero(nl_cn_var)]
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
plt.scatter(nl_cn_var,nl_fn_var,color=[0,0,0],s=8)
plt.plot(np.arange(0,1,.01),np.arange(0,1,.01)*fn_cn_k+popt[1],'r-')
plt.text(0,70,'R = ' + str(fn_cn_R) + ', k = ' + str(np.round(fn_cn_k,2)))
plt.ylabel('PC Space Functional Distance')
plt.xlabel('Reiser Optimal Infrac Band Distance')

# cluster on connection distance, plot physiology distance by sorted cnx clusts
# CLUSTER BLOCK
# define matrix to cluster on
input_matrix = thresh_optimal_bw_dist_mat
input_labels = thresh_phys_dist_types
mirror_matrix = thresh_pc_dists_table
# train and fit the KMeans clustering model
clust_model = KMeans(n_clusters=nclusts, random_state=0)
clust_IDs = clust_model.fit_predict(input_matrix)
# reorganize the distances matrix by cluster. within each cluster, sort cell types by mean distance, from closest to furthest
resorted_data = np.zeros(input_matrix.shape[0]).reshape(1,-1)
resorted_mirror = np.zeros(mirror_matrix.shape[0]).reshape(1,-1)
resorted_types = np.full(1,'',dtype='<U32')
for clust in range(0,np.max(clust_IDs)+1):
    clust_sortinds = np.argsort(np.mean(np.abs(input_matrix[clust_IDs==clust,:]),axis=1))
    resorted_mirror = np.append(resorted_mirror, mirror_matrix[clust_IDs==clust,:][clust_sortinds,:], axis=0)
    resorted_data = np.append(resorted_data, input_matrix[clust_IDs==clust,:][clust_sortinds,:], axis=0)
    resorted_types = np.append(resorted_types, input_labels[clust_IDs==clust][clust_sortinds])
# trim initialization axis
resorted_mirror = resorted_mirror[1:,:]
resorted_data = resorted_data[1:,:]
resorted_types = resorted_types[1:]
# repeat to sort matrix columns
reresorted_data = np.zeros(resorted_data.shape[0]).reshape(-1,1)
reresorted_mirror = np.zeros(resorted_mirror.shape[0]).reshape(-1,1)
for clust in range(0,np.max(clust_IDs)+1):
    clust_sortinds = np.argsort(np.mean(np.abs(resorted_data[:,clust_IDs==clust]),axis=0))
    reresorted_mirror = np.append(reresorted_mirror, resorted_mirror[:,clust_IDs==clust][:,clust_sortinds], axis=1)
    reresorted_data = np.append(reresorted_data, resorted_data[:,clust_IDs==clust][:,clust_sortinds], axis=1)
reresorted_data = reresorted_data[:,1:]
reresorted_mirror = reresorted_mirror[:,1:]
# plotting
plt.subplot(3,3,4)
plt.title('Reiser Optimal Infrac Band Dist, nclust =' + str(np.max(clust_IDs)+1))
x = plt.imshow(reresorted_data, cmap='cividis', clim=[0,0.6])
plt.colorbar(shrink=0.5,aspect=8,ticks=[0,0.6])
# set ticks
x.axes.get_yaxis().set_ticks([])
x.axes.get_xaxis().set_ticks([])
plt.subplot(3,3,5)
plt.title('Cluster-mirrored func distance')
x = plt.imshow(reresorted_mirror, cmap='pink', clim=[10,80])
plt.colorbar(shrink=0.5,aspect=8,ticks=[10,45,80])
# set ticks
x.axes.get_yaxis().set_ticks([])
x.axes.get_xaxis().set_ticks([])

# cluster on connection distance, plot physiology distance by sorted cnx clusts
# CLUSTER BLOCK
# define matrix to cluster on
input_matrix = thresh_pc_dists_table
input_labels = thresh_phys_dist_types
mirror_matrix = thresh_optimal_bw_dist_mat
# train and fit the KMeans clustering model
clust_model = KMeans(n_clusters=nclusts, random_state=0)
clust_IDs = clust_model.fit_predict(input_matrix)
# reorganize the distances matrix by cluster. within each cluster, sort cell types by mean distance, from closest to furthest
resorted_data = np.zeros(input_matrix.shape[0]).reshape(1,-1)
resorted_mirror = np.zeros(mirror_matrix.shape[0]).reshape(1,-1)
resorted_types = np.full(1,'',dtype='<U32')
for clust in range(0,np.max(clust_IDs)+1):
    clust_sortinds = np.argsort(np.mean(np.abs(input_matrix[clust_IDs==clust,:]),axis=1))
    resorted_mirror = np.append(resorted_mirror, mirror_matrix[clust_IDs==clust,:][clust_sortinds,:], axis=0)
    resorted_data = np.append(resorted_data, input_matrix[clust_IDs==clust,:][clust_sortinds,:], axis=0)
    resorted_types = np.append(resorted_types, input_labels[clust_IDs==clust][clust_sortinds])
# trim initialization axis
resorted_mirror = resorted_mirror[1:,:]
resorted_data = resorted_data[1:,:]
resorted_types = resorted_types[1:]
# repeat to sort matrix columns
reresorted_data = np.zeros(resorted_data.shape[0]).reshape(-1,1)
reresorted_mirror = np.zeros(resorted_mirror.shape[0]).reshape(-1,1)
for clust in range(0,np.max(clust_IDs)+1):
    clust_sortinds = np.argsort(np.mean(np.abs(resorted_data[:,clust_IDs==clust]),axis=0))
    reresorted_mirror = np.append(reresorted_mirror, resorted_mirror[:,clust_IDs==clust][:,clust_sortinds], axis=1)
    reresorted_data = np.append(reresorted_data, resorted_data[:,clust_IDs==clust][:,clust_sortinds], axis=1)
reresorted_data = reresorted_data[:,1:]
reresorted_mirror = reresorted_mirror[:,1:]
# plotting
plt.subplot(3,3,7)
plt.title('Reiser Optimal Infrac Band Dist, nclust =' + str(np.max(clust_IDs)+1))
x = plt.imshow(reresorted_mirror, cmap='cividis', clim=[0,0.6])
plt.colorbar(shrink=0.5,aspect=8,ticks=[0,0.6])
# set ticks
x.axes.get_yaxis().set_ticks([])
x.axes.get_xaxis().set_ticks([])
plt.subplot(3,3,8)
plt.title('Cluster-mirrored func distance')
x = plt.imshow(reresorted_data, cmap='pink', clim=[10,80])
plt.colorbar(shrink=0.5,aspect=8,ticks=[10,45,80])
# set ticks
x.axes.get_yaxis().set_ticks([])
x.axes.get_xaxis().set_ticks([])

# plot distribution of connectivity distances for all conn vs optimal subset
plt.subplot(3,3,9)
h,x = np.histogram(thresh_Rsr_allcnx_dist_mat[np.nonzero(thresh_Rsr_allcnx_dist_mat)], bins=66, range=(0,1))
plt.plot(x[1:],h/np.sum(h))
h,x = np.histogram(thresh_optimal_bw_dist_mat[np.nonzero(thresh_optimal_bw_dist_mat)], bins=66, range=(0,1))
plt.plot(x[1:],h/np.sum(h))

# save it
fname = '/Users/tcurrier/Desktop/Clandinin Lab/Imaging/medulla project/all cell summaries/cnx-phys Reiser all inputs vs. optimal subset.pdf'
similarityfig.savefig(fname, format='pdf', orientation='landscape')




#%% ONE-TIME ONLY BLOCKS BELOW:

#%% parse Reiser edgelist into type-to-type matrix - this only needs to be run once, then can be opened via the saved .npy files

# open full edge list
Rsr_edgelist = []
with open('/Users/tcurrier/Desktop/Clandinin Lab/Connectomics/NernReiser2024 Data/adj_ol_with_roi.csv', mode='r') as f:
    reader = csv.reader(f)
    for row in reader:
        Rsr_edgelist.append(row)

# convert to array and discard cell body ID information to speed up array searches
Rsr_edgelist = np.asarray(Rsr_edgelist)
Rsr_edgelist = Rsr_edgelist[:,2:]

# create the container type to type matrix based on the list of unique cell types
typelist = np.unique(Rsr_edgelist[1:,2])
Rsr_type2type = np.zeros((len(typelist),len(typelist)))

# define the list of acceptable ROIs to use for weight calculation (only count connections in OL)
OL_roilist = np.asarray(['AME(L)', 'AME(R)', 'LA(R)', 'LO(L)', 'LO(R)', 'LOP(L)', 'LOP(R)', 'ME(L)', 'ME(R)', 'Optic-unspecified(L)', 'Optic-unspecified(R)'])

# loop over each row in the edgelist and add weight to the appropriate index in the type2type matrix
for ind in range(0,Rsr_edgelist.shape[0]):
    if Rsr_edgelist[ind,0] in OL_roilist:
        current_edge = Rsr_edgelist[ind,:]
        pre_ind = np.where(typelist==current_edge[2])[0][0]
        post_ind = np.where(typelist==current_edge[3])[0][0]
        w_to_add = current_edge[1].astype('int')
        Rsr_type2type[pre_ind,post_ind] = Rsr_type2type[pre_ind,post_ind] + w_to_add

# create a vector that will contain compile flags to mark deletion status
compile_flag = np.zeros(len(typelist))

# loop over types and, if a '_L' tag is found, compile with '_R' row, if present
for type in range(0,len(typelist)):
    # check for L tag
    if typelist[type].split('_')[1] == 'L':
        # check if the R-tagged version of the same cell type is also in the list
        if (typelist[type].split('_')[0] + '_R') in typelist:
            # if it is, add those rows together, this is equivalent to pooling INPUT ONLY - we will discard the output side information, as we only want to consider the inputs in a single optic lobe
            Rsr_type2type[typelist == (typelist[type].split('_')[0] + '_R'),:] = Rsr_type2type[typelist == (typelist[type].split('_')[0] + '_R'),:] + Rsr_type2type[typelist == (typelist[type].split('_')[0] + '_L'),:]
            # flag this index for deletion
            compile_flag[type] = 1
        else:
            # if there is no corresponding R-flagged type (i.e., a cell type only give contralateral output), DO NOT mark this cell type for deletion
            pass

# after adding all L-tag weights to their R-tag counterparts, delete all compiled rows
Rsr_type2type = np.delete(np.delete(Rsr_type2type, np.nonzero(compile_flag), axis=0), np.nonzero(compile_flag), axis=1)
typelist = np.delete(typelist, np.nonzero(compile_flag), axis=0)
del compile_flag

# now that we have compressed each cell type into a single instance (mostly R's, but some L's for contralateral-only projecting cell types), loop over cell types in the list and cut the R or L tags
for type in range(0,len(typelist)):
    typelist[type] = typelist[type].split('_')[0]

# divide each column by the sum of all counts for that column, converts raw counts into input fractions!
Rsr_type2type_infrac = np.zeros(Rsr_type2type.shape)
for post_type in range(0,Rsr_type2type.shape[1]):
    current_inputs = Rsr_type2type[:,post_type]
    Rsr_type2type_infrac[:,post_type] = current_inputs/np.sum(current_inputs)

#%% save raw and total input normalized versions of the table

np.save('/Users/tcurrier/Desktop/Clandinin Lab/Connectomics/NernReiser2024 Data/__EXTRACTED__type2type_counts.npy', Rsr_type2type)
np.save('/Users/tcurrier/Desktop/Clandinin Lab/Connectomics/NernReiser2024 Data/__EXTRACTED__type2type_input_fractions.npy', Rsr_type2type_infrac)
np.save('/Users/tcurrier/Desktop/Clandinin Lab/Connectomics/NernReiser2024 Data/__EXTRACTED__cell_types.npy', typelist)

#%% compute type-to-type Jaccard similarity scores based on binarized Reiser data (or weighted Jaccard similarity for raw counts data)
# this only needs to be run once, then can be passed to ME_temporal_connectome_correlate in place of FAFB data

# open data
Rsr_conn_table = np.load('/Users/tcurrier/Desktop/Clandinin Lab/Connectomics/NernReiser2024 Data/__EXTRACTED__type2type_counts.npy')

# binarize conn table
binary_conn_table = np.where(Rsr_conn_table > 0, 1, 0)
# append transpose of binary table to create binary all cnx table
binary_inout_conn_table = np.append(binary_conn_table, binary_conn_table.T, axis=0)

# append transpose of counts table to create binary all cnx counts table
counts_conn_table = Rsr_conn_table
counts_out_conn_table = counts_conn_table.T
counts_inout_conn_table = np.append(counts_conn_table, counts_conn_table.T, axis=0)

# retain only columns (post-synaptic types) that are present in phys_dist_types. keep all rows.
binary_conn_subset = np.zeros((binary_conn_table.shape[0],len(phys_dist_types)))
binary_inout_conn_subset = np.zeros((binary_inout_conn_table.shape[0],len(phys_dist_types)))
counts_conn_subset = np.zeros((counts_conn_table.shape[0],len(phys_dist_types)))
counts_out_conn_subset = np.zeros((counts_out_conn_table.shape[0],len(phys_dist_types)))
counts_inout_conn_subset = np.zeros((counts_inout_conn_table.shape[0],len(phys_dist_types)))
for n in range(0,len(phys_dist_types)):
    current_type = phys_dist_types[n]
    if current_type == 'T4':
        current_type = 'T4a'
    binary_conn_subset[:,n] = binary_conn_table[:,Rsr_types==current_type].reshape(-1)
    binary_inout_conn_subset[:,n] = binary_inout_conn_table[:,Rsr_types==current_type].reshape(-1)
    counts_conn_subset[:,n] = counts_conn_table[:,Rsr_types==current_type].reshape(-1)
    counts_out_conn_subset[:,n] = counts_out_conn_table[:,Rsr_types==current_type].reshape(-1)
    counts_inout_conn_subset[:,n] = counts_inout_conn_table[:,Rsr_types==current_type].reshape(-1)

# create placeholder matrices for allcnx and incnx jaccard distances
Rsr_allcnx_Jdist_mat = np.zeros((len(phys_dist_types),len(phys_dist_types)))
Rsr_incnx_Jdist_mat = np.zeros((len(phys_dist_types),len(phys_dist_types)))
Rsr_allcnx_wJdist_mat = np.zeros((len(phys_dist_types),len(phys_dist_types)))
Rsr_incnx_wJdist_mat = np.zeros((len(phys_dist_types),len(phys_dist_types)))
Rsr_outcnx_wJdist_mat = np.zeros((len(phys_dist_types),len(phys_dist_types)))

# calculate type-to-type Jaccard distances based on inputs vectors
for ind1 in range(0,len(phys_dist_types)):
    first_label = phys_dist_types[ind1]
    v1 = binary_conn_subset[:,phys_dist_types == first_label]
    w1 = binary_inout_conn_subset[:,phys_dist_types == first_label]
    a1 = counts_conn_subset[:,phys_dist_types == first_label]
    b1 = counts_inout_conn_subset[:,phys_dist_types == first_label]
    c1 = counts_out_conn_subset[:,phys_dist_types == first_label]
    for ind2 in range(ind1+1,len(phys_dist_types)):
        second_label = phys_dist_types[ind2]
        v2 = binary_conn_subset[:,phys_dist_types == second_label]
        w2 = binary_inout_conn_subset[:,phys_dist_types == second_label]
        a2 = counts_conn_subset[:,phys_dist_types == second_label]
        b2 = counts_inout_conn_subset[:,phys_dist_types == second_label]
        c2 = counts_out_conn_subset[:,phys_dist_types == second_label]
        # Jaccard distance is 1 - (the # of "1" rows that overlap in the two vectors divided by the total number of "1" rows across both vectors)
        Rsr_incnx_Jdist_mat[ind1,ind2] = 1 - (len(np.where(np.logical_and(v1 == 1, v2 == 1).reshape(-1))[0]) / len(np.where(np.logical_or(v1 == 1, v2 == 1).reshape(-1))[0]))
        Rsr_allcnx_Jdist_mat[ind1,ind2] = 1 - (len(np.where(np.logical_and(w1 == 1, w2 == 1).reshape(-1))[0]) / len(np.where(np.logical_or(w1 == 1, w2 == 1).reshape(-1))[0]))
        # weighted Jaccard distance is 1 - (sum of all smallest values between v1 and v2 divided by the sum of all largest values between v1 and v2)
        Rsr_incnx_wJdist_mat[ind1,ind2] = 1 - (np.sum(np.min(np.append(a1,a2,axis=1),axis=1)) / np.sum(np.max(np.append(a1,a2,axis=1),axis=1)))
        Rsr_allcnx_wJdist_mat[ind1,ind2] = 1 - (np.sum(np.min(np.append(b1,b2,axis=1),axis=1)) / np.sum(np.max(np.append(b1,b2,axis=1),axis=1)))
        Rsr_outcnx_wJdist_mat[ind1,ind2] = 1 - (np.sum(np.min(np.append(c1,c2,axis=1),axis=1)) / np.sum(np.max(np.append(c1,c2,axis=1),axis=1)))


# reflect across the main diagonal to make table lookup easier in the weighting steps
lower_inds = np.nonzero(np.tril(np.ones((len(phys_dist_types),len(phys_dist_types)))))
Rsr_incnx_Jdist_mat[lower_inds] = Rsr_incnx_Jdist_mat.T[lower_inds]
Rsr_allcnx_Jdist_mat[lower_inds] = Rsr_allcnx_Jdist_mat.T[lower_inds]
Rsr_incnx_wJdist_mat[lower_inds] = Rsr_incnx_wJdist_mat.T[lower_inds]
Rsr_allcnx_wJdist_mat[lower_inds] = Rsr_allcnx_wJdist_mat.T[lower_inds]
Rsr_outcnx_wJdist_mat[lower_inds] = Rsr_outcnx_wJdist_mat.T[lower_inds]


#%% save these Jaccard distance matrices
np.save('/Users/tcurrier/Desktop/Clandinin Lab/Connectomics/NernReiser2024 Data/Rsr_distances_culled.npy', Rsr_allcnx_Jdist_mat)
np.save('/Users/tcurrier/Desktop/Clandinin Lab/Connectomics/NernReiser2024 Data/Rsr_distances_inputs_only_culled.npy', Rsr_incnx_Jdist_mat)
np.save('/Users/tcurrier/Desktop/Clandinin Lab/Connectomics/NernReiser2024 Data/Rsr_weighted_distances_culled.npy', Rsr_allcnx_wJdist_mat)
np.save('/Users/tcurrier/Desktop/Clandinin Lab/Connectomics/NernReiser2024 Data/Rsr_weighted_distances_inputs_only_culled.npy', Rsr_incnx_wJdist_mat)
np.save('/Users/tcurrier/Desktop/Clandinin Lab/Connectomics/NernReiser2024 Data/Rsr_weighted_distances_outputs_only_culled.npy', Rsr_outcnx_wJdist_mat)
np.save('/Users/tcurrier/Desktop/Clandinin Lab/Connectomics/NernReiser2024 Data/Rsr_distances_types.npy', phys_dist_types)
