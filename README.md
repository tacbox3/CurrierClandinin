Code base for "Infrequent strong connections constrain connectomic predictions of neuronal function" by TA Currier and TR Clandinin, 2025. Please direct all questions or comments to Tim (currier@stanford.edu).

# Contents
## Connectome Data
- Filenames that begin with `__EXTRACTED__` are the full type-to-type connectivity data from Nern et al., *Nature*, 2025 - used for calculating Euclidean connectivity distance
- Filenames that begin with `Rsr_` are type-to-type weighted Jaccard distance matrices for the Nern et al. data
- `complete_metrics` courtesy of the Reiser Lab
- `FAFB_distances` courtesy of Sebastian Seung - type-to-type weighted Jaccard distances for flywire data, Matsliah et al., *Nature*, 2024
- `HighN_InFracs...` scraped from the Reiser Lab's [Cell Type Explorer Website](https://reiserlab.github.io/male-drosophila-visual-system-connectome/)

## Logs
- `ME0708_full_log_snap.xls` is the main ROI metadata log file, contains fly, date, cell type, and stimulus information for each ROI in the dataset - most scripts in the top level folder use this log as the main source of metadata
- `ME0708_proofed_culled_log` is a secondary log with all "incomplete data" cells removed (i.e., missing a noise color) - used by the PCA script, which doesn't handle NaNs well

##
