import sys, os
import math
import numpy as np
import matplotlib.pyplot as plt
from nlb_tools.nwb_interface import NWBDataset
import pandas as pd
import subprocess
from scipy.ndimage import gaussian_filter1d


from neo.core import AnalogSignal
import quantities as pq
from elephant import spike_train_generation

from pathlib import Path
import configparser

properties = configparser.ConfigParser()
properties.read('gen.params')

save_dir = properties.get('path', 'save_dir')
data_dir = properties.get('path', 'data_dir')

target_cell_num = properties.getint('dataset', 'target_cell_num')
fr_scale = properties.getint('dataset', 'fr_scale')

Path(save_dir).mkdir(parents=True, exist_ok=True)

original_num = 182 # for the dandi MAZE dataset
duplicate_num = (target_cell_num - 1) // original_num + 1

if fr_scale > 1:
    dat = np.load(save_dir + "/spikes_{}_duplicate_{}_{}x.npz".format("MAZE", duplicate_num, fr_scale))
else:
    dat = np.load(save_dir + "/spikes_{}_duplicate_{}.npz".format("MAZE", duplicate_num))
X_train             = dat["X_train"]
X_valid             = dat["X_valid"]

filter_list = [40]
gaussian_spikes_train_list = {}
gaussian_spikes_valid_list = {}
for filter_dat in filter_list:
    gaussian_spikes_train_list[filter_dat] = []
    gaussian_spikes_valid_list[filter_dat] = []

print(X_train.shape)
for trial_idx in range(X_train.shape[0]):
    for filter_dat in filter_list:
        generated_spikes = np.zeros((X_train.shape[2], X_train.shape[1]))
        for elec in range(X_train.shape[2]):
            spike_list = X_train[trial_idx,:,elec]
            filter = gaussian_filter1d(spike_list.astype('float32'), filter_dat)
            generated_spikes[elec,:] = filter
        generated_spikes = np.transpose(generated_spikes)
        gaussian_spikes_train_list[filter_dat].append(generated_spikes)

for filter_dat in filter_list:
    gaussian_spikes_train_list[filter_dat] = np.asarray(gaussian_spikes_train_list[filter_dat])
    print(gaussian_spikes_train_list[filter_dat].shape)

print(X_valid.shape)
for trial_idx in range(X_valid.shape[0]):
    for filter_dat in filter_list:
        generated_spikes = np.zeros((X_valid.shape[2], X_valid.shape[1]))
        for elec in range(X_valid.shape[2]):
            spike_list = X_valid[trial_idx,:,elec]
            filter = gaussian_filter1d(spike_list.astype('float32'), filter_dat)
            generated_spikes[elec,:] = filter
        generated_spikes = np.transpose(generated_spikes)
        gaussian_spikes_valid_list[filter_dat].append(generated_spikes)

for filter_dat in filter_list:
    gaussian_spikes_valid_list[filter_dat] = np.asarray(gaussian_spikes_valid_list[filter_dat])
    print(gaussian_spikes_valid_list[filter_dat].shape)

print('FILTER DONE')

for filter_dat in filter_list:
    if fr_scale > 1:
        gauss_file_name = "/spikes_{}_duplicate_{}_{}x_gaussian_{}.npz".format("MAZE", duplicate_num, fr_scale, filter_dat)
    else:
        gauss_file_name = "/spikes_{}_duplicate_{}_gaussian_{}.npz".format("MAZE", duplicate_num, filter_dat)
    np.savez(save_dir + gauss_file_name, X_train_gaussian = gaussian_spikes_train_list[filter_dat],
                                         X_valid_gaussian = gaussian_spikes_valid_list[filter_dat])
