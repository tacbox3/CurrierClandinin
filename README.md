Code base for "Infrequent strong connections constrain connectomic predictions of neuronal function" by TA Currier and TR Clandinin, *Cell*, 2025. Please direct all questions or comments to Tim (currier@stanford.edu). Readers seeking to download visual response data for all recorded cell types should download `ALL_RESPONSES.npy` [here](10.5061/dryad.pg4f4qs1j).

# Contents

## Logs
- `ME0708_full_log_snap.xls` is the main ROI metadata log file, contains fly, date, cell type, and stimulus information for each ROI in the dataset - most scripts in the top level folder use this log as the main source of metadata
- `ME0708_proofed_culled_log` is a secondary log with all "incomplete data" cells removed (i.e., missing a noise color) - used by the PCA script, which doesn't handle NaNs well

## tac_util
A few helper functions for interacting with .hdf5 metadata files

## Scripts
- The core analysis code for the manuscript lives in the scripts beginning with `ME_`. Comments at the beginning of each script detail its function. Please note that some scripts will require the connectome data deposited with the physiology data on Dryad (see **Raw Data Availability**, below).
- `ALL_RESPONSES.npy` (available to download on Dryad) contains responses generated via sequential runs of `ME_import_save.py` and `ME_load_center.py`. Users not wishing to re-analyze all raw imaging data are encouraged to work from `ALL_RESPONSES.npy`. Note that this dictionary will need to be broken into standalone variables for the subsequent code to run (i.e., `ME_analyze_plt.py`).
- `recover_isotropy.py` and `var_brain.py` are not run in any `ME_` scripts, but are included for completeness - these are run as part of the preprocessing pipeline that also includes motion correction (performed with [brainsss](https://github.com/ClandininLab/brainsss))
- `save_strfs.py` is a callable script that creates STRF movies for a given recording series
- **WARNING!** The path structure of the included scripts are **not** internally consistent. All paths will need to be redefined by the user based on the location of each programmatic element (raw data, scripts, logs, connectome data, and saved .npy summary variables)

# Required packages
The following packages are required to run the scripts:
- [Visanalysis](https://github.com/ClandininLab/visanalysis)
- scipy
- sklearn
- skimage
- cv2
- csv
- xlrd
- matplotlib

# Raw Data Availability
Raw Data, including STRFs for all recorded neurons, can be downloaded on Dryad at the following DOIs. The ImagingData files available for download have been minimally preprocessed (see **Scripts**, above)
- [10.5061/dryad.pg4f4qs1j]() *(this is the main deposition)*
- [10.5061/dryad.bnzs7h4ns]()
- [10.5061/dryad.kh18932k1]()
