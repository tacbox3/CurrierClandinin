Code base for "Infrequent strong connections constrain connectomic predictions of neuronal function" by TA Currier and TR Clandinin, 2025. Please direct all questions or comments to Tim (currier@stanford.edu).

# Contents

## PhysiologySummary 
`ALL_RESPONSES.npy` is intended to be an easy-to-find compendium of all cells recorded for the study. The last dimension of each dict value is 571, the number of ROIs. The dict contains the following keys:
  * `'cell_IDs'`, the unique cell identifiers (ROIs with the same cell_ID were recorded from the same cell)
  * `'cell_types'`, the cell type assignment for each ROI (empty elements indicate that no identification was made)
  * `'blue_STRFs'`, the STRF for each ROI responding to blue noise (dims: x,y,t,ROI#)
  * `'uv_STRFs'`, the STRF for each ROI responding to UV noise (dims: x,y,t,ROI#)
  * `'STRF_time`, the time vector for STRF data
  * `'flicker_responses'`, the dF/F response for each ROI responding to full-screen flicker at 0.1, 0.5, 1 or 2 Hz (dims: f,t,ROI#)
  * `'flicker_time'`, the time vector for flicker data

## Connectome Data
- `complete_metrics` courtesy of the Reiser Lab, summary quantifications of neuronal morphology
- Filenames that begin with `__EXTRACTED__` are the full type-to-type connectivity data from Nern et al., *Nature*, 2025 - used for calculating Euclidean connectivity distance
- Filenames that begin with `Rsr_` are type-to-type weighted Jaccard distance matrices for the Nern et al. data
- `FAFB_distances` courtesy of Sebastian Seung - type-to-type weighted Jaccard distances for flywire data, Matsliah et al., *Nature*, 2024
- `HighN_InFracs...` scraped from the Reiser Lab's [Cell Type Explorer Website](https://reiserlab.github.io/male-drosophila-visual-system-connectome/)

## Logs
- `ME0708_full_log_snap.xls` is the main ROI metadata log file, contains fly, date, cell type, and stimulus information for each ROI in the dataset - most scripts in the top level folder use this log as the main source of metadata
- `ME0708_proofed_culled_log` is a secondary log with all "incomplete data" cells removed (i.e., missing a noise color) - used by the PCA script, which doesn't handle NaNs well

## tac_util
A few helper functions for interacting with .hdf5 metadata files

## Scripts
- The core analysis code for the manuscript lives in the scripts beginning with `ME_`
- `recover_isotropy.py` and `var_brain.py` are not run in any `ME_` scripts, but are included for completeness - these are run as part of the preprocessing pipeline that also includes motion correction (performed with [brainsss](https://github.com/ClandininLab/brainsss))
- `save_strfs.py` is a callable script that can be used to create STRF movies for a given recording series - the raw data must be downloaded to create these movies
- **WARNING!** The path structure of the included scripts are **not** internally consistent, and will need to be renamed based on the location you put each element (raw data, scripts, logs, connectome data, saved .npy summary variables)

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
Raw Data can be downloaded on Dryad at the following DOIs. The ImagingData files available for download have been preprocessed (see **Scripts**, above)
- [10.5061/dryad.pg4f4qs1j]() *(this is the main deposition)*
- [10.5061/dryad.bnzs7h4ns]()
- [10.5061/dryad.kh18932k1]()
