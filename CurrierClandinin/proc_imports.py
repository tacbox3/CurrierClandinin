
from visanalysis import plugin
import sys
import os
import shutil
import glob
from pathlib import Path

import_dir = sys.argv[1]
# name of import directory in imports folder, e.g. 20200803-mht

data_directory = '/oak/stanford/groups/trc/data/Tim/ImagingData/'

# (1) COPY TO NEW DATE DIRECTORY
from_import_directory = os.path.join(data_directory, 'imports', import_dir)
output_subdir = import_dir.split('-')[0]
#format is yyyymmdd, remove any tag or suffix, e.g. '-mht'
new_imaging_directory = os.path.join(data_directory, 'processed', output_subdir)
Path(new_imaging_directory).mkdir(exist_ok=True)
#make new directory for this date
print('Made directory {}'.format(new_imaging_directory))

for subdir in os.listdir(from_import_directory): # one subdirectory per series
    current_timeseries_directory = os.path.join(from_import_directory, subdir)
    for fn in glob.glob(os.path.join(current_timeseries_directory, 'T*')):
        dest = os.path.join(new_imaging_directory, os.path.split(fn)[-1])
        shutil.copyfile(fn, dest)

# (2) ATTACH VISPROTOCOL DATA
# Make a backup of raw visprotocol datafile before attaching data to it
experiment_file_name = '{}-{}-{}.hdf5'.format(output_subdir[0:4], output_subdir[4:6], output_subdir[6:8])
experiment_filepath = os.path.join(data_directory, 'DataFiles', experiment_file_name)
shutil.copy(experiment_filepath, os.path.join(data_directory, 'RawDataFiles', experiment_file_name))

plug = plugin.bruker.BrukerPlugin()
plug.attachData(experiment_file_name.split('.')[0], experiment_filepath, new_imaging_directory)

print('Attached data to {}'.format(experiment_filepath))
