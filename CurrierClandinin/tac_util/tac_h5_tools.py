# tag_h5.py
# author: Tim Currier; last updated: 2022-11-02
# module containing functions related to H5 file tags for cell type information

# import relevant packages
import functools
import os
import sys
import h5py
import numpy as np
from visanalysis.util import h5io

# Opens the specified HDF5 file and adds the provided tags to the specified ROI. All arguments must be provided, with unidentified cells marked as None for the 5th argument. series_number, roi_number, and cert are integers, all other arguments are strings. cell_num must be a string with 4 digits; i.e. 0210.
# argumenmts: [1] H5 file path; [2] series_number; [3] roi_set_name; [4] roi_number; [5] cell_class; [6] cell_type; [7] certainty; [8] ('alt_type1','alt_type2',...) [9] cell_num
# implementation: add_type_tag('/Volumes/TimBigData/Bruker/StimData/2022-10-20.hdf5', 1, 'roi_set_blue', 0, 'Tm3', 8, ('Tm3Y','TmY3'), '0120'):
def add_type_tag(file_path, series_number, roi_set, roi_number, cell_class, cell_type, cert, alt_types, cell_num):
    # open HDF5 file and navigate to specified ROI
    with h5py.File(file_path, 'r+') as experiment_file:
        find_partial = functools.partial(h5io.find_series, sn=series_number)
        epoch_run_group = experiment_file.visititems(find_partial)
        # flag the series as containing good ROIs
        epoch_run_group.attrs['analyze_bool'] = True
        parent_roi_group = epoch_run_group.get('rois')
        roi_group = parent_roi_group.get(roi_set)
        roi_path = roi_group.get('roipath_{}'.format(str(roi_number)))
        # flag ROI as ID'd if a cell type is given
        roi_path.attrs['cell_class'] = cell_class
        if cell_type is not None:
            roi_path.attrs['id_bool'] = True
            # add tags
            roi_path.attrs['cell_type'] = cell_type
            roi_path.attrs['alt_types'] = alt_types
            roi_path.attrs['certainty'] = cert
        else:
            roi_path.attrs['id_bool'] = False
        # pull fly ID for use in cell_id generation
        fly_group = parent_roi_group.parent.parent.parent
        fly_id = fly_group.attrs['fly_id']
        roi_path.attrs['cell_id'] = fly_id+'-'+cell_num

# Pulls the 'frame_times' data object in a series' H5 file, then takes the difference between the first and last entries to calculate the lag associated with each z-slice.
# argumenmts: [1] H5 file path; [2] series_number
# implementation: frame_offsets = get_vol_frame_offsets('/Volumes/TimBigData/Bruker/StimData/2022-10-20.hdf5', 1)
def get_vol_frame_offsets(file_path, series_number):
    with h5py.File(file_path, 'r') as experiment_file:
        find_partial = functools.partial(h5io.find_series, sn=series_number)
        series_group = experiment_file.visititems(find_partial)
        acquisition_group = series_group.get('acquisition')
        frame_times = np.asarray(acquisition_group.get('frame_times'))
    # initialize frame_offsets and populate
    frame_offsets = np.zeros((frame_times.shape[1],1))
    for frame_ind in range(0,frame_times.shape[1]):
        # offsets are consitent for every single volume, so we only need to do this calculation once for each z-slice
        frame_offsets[frame_ind] = frame_times[0,frame_ind]-frame_times[0,0]
    return frame_offsets

# get_roi_zslice has been removed due to poor backwards compatability
