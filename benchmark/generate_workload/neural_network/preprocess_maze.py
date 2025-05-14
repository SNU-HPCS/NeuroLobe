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
sampling_rate = 1000  # for the dandi MAZE dataset
duplicate_num = (target_cell_num - 1) // original_num + 1

print("Downloading the dataset ...")
# Load dataset
if not os.path.exists(data_dir):
    proc = subprocess.Popen(['dandi download https://gui.dandiarchive.org/#/dandiset/000128'], close_fds=True, shell=True, cwd=".")
    proc.communicate()
    proc.wait()

    proc = subprocess.Popen(['mv 000128 ../spike_generator/dandi_data/.'], close_fds=True, shell=True, cwd=".")
    proc.communicate()
    proc.wait()


print("Parsing the dataset ...")
dataset = NWBDataset(data_dir, split_heldout=False)

# Extract neural data from the NWB dataset
lag = 100
trial_start = -250
trial_end = 450
trial_dur = trial_end - trial_start
trial_data = dataset.make_trial_data(align_field='move_onset_time', align_range =(trial_start, trial_end))
grouped_trial_data = list(trial_data.groupby('trial_id', sort=False))

spikes_list = []
hand_vel_list = []

print("Converting the Dataset")
sys.stdout.flush()

# We decode the hand velocity for MAZE and BUMP dataset
# and decode the finger velocity for RTT dataset
vel_name = "hand_vel"

train_list = []
valid_list = []
trial_id = 0
for trial_idx in range(len(grouped_trial_data)):
    trial_info = dataset.trial_info.split[trial_idx]

    if trial_info == 'train': train_list.append(trial_id)
    if trial_info == 'val': train_list.append(trial_id)
    if trial_info == 'test': valid_list.append(trial_id)

    spike_dat = grouped_trial_data[trial_id][1]['spikes'].to_numpy()
    spike_dat = spike_dat[:-lag,:]
    if duplicate_num == 1:
        spikes_list.append(spike_dat)
    else:
        # Retrieve average firing rate using gaussian filter
        # Averate the filter using
        gaussian_spike = np.zeros(spike_dat.shape)

        # Generate new shape
        new_shape = (spike_dat.shape[0], spike_dat.shape[1] * duplicate_num)
        generated_spikes = np.zeros(new_shape)


        for elec in range(spike_dat.shape[1]):
            gaussian_spike[:,elec] = gaussian_filter1d(spike_dat[:,elec].astype('float32'), 100000 / sampling_rate)
            # To calculate the instant rate
            gaussian_spike[:,elec] = gaussian_spike[:,elec] * sampling_rate
            # Retrieve analog signal
            analog_sig = AnalogSignal(gaussian_spike[:,elec], units='Hz', sampling_rate=sampling_rate*pq.Hz, refractory=1000/sampling_rate*pq.ms)
            spike_generator = spike_train_generation.NonStationaryPoissonProcess(analog_sig)
            for dup_id in range(duplicate_num):
                spike_time = spike_generator._generate_spiketrain_as_array()
                # Generate random spike times
                spike_time = spike_time * sampling_rate
                spike_time = np.array(spike_time, dtype=int)
                spike_list = np.zeros(spike_dat.shape[0], dtype=int)
                for spk_time in spike_time:
                    spike_list[spk_time] = 1
                
                generated_spikes[:,elec * duplicate_num + dup_id] = spike_list
        
        spikes_list.append(generated_spikes)

    hand_vel_dat = grouped_trial_data[trial_id][1][vel_name].to_numpy()
    hand_vel_dat = hand_vel_dat[lag:,:]
    hand_vel_list.append(hand_vel_dat)
    trial_id += 1

spikes_list = np.asarray(spikes_list)
hand_vel_list = np.asarray(hand_vel_list)

# print(spikes_list.shape)
# print(hand_vel_list.shape)

if fr_scale > 1:
    original_spikes_list = spikes_list
    print(spikes_list.shape)
    spikes_list = spikes_list[:, :(spikes_list.shape[1] - spikes_list.shape[1] % fr_scale), :]
    hand_vel_list = hand_vel_list[:, :(hand_vel_list.shape[1] - hand_vel_list.shape[1] % fr_scale), :]
    print("\nScaling firing rate by {}x".format(fr_scale))
    spikes_list = spikes_list.reshape(spikes_list.shape[0], -1, fr_scale, spikes_list.shape[2]).sum(axis=2)
    hand_vel_list = hand_vel_list.reshape(spikes_list.shape[0], -1, fr_scale, hand_vel_list.shape[2]).sum(axis=2)

    print("firing rate: {:.2f} -> {:.2f}".format(np.sum(original_spikes_list / original_spikes_list.shape[0] / original_spikes_list.shape[1] / original_spikes_list.shape[2], axis=(0,1,2)) * sampling_rate, \
           np.sum(spikes_list / spikes_list.shape[0] / spikes_list.shape[1] / spikes_list.shape[2], axis=(0,1,2)) * sampling_rate))


print("Binning the dataset ...")

X_train       = {}
X_train_spike = {}
y_train       = {}
X_valid       = {}
X_valid_spike = {}
y_valid       = {}

X             = {}
X_spike       = {}
y             = {}

neural_data = spikes_list
hand_vel_data = hand_vel_list
spike_data = (neural_data > 0).astype(float)

X = neural_data
X_spike = spike_data
y = hand_vel_data

neural_data_train = neural_data[train_list]
spike_data_train = spike_data[train_list]
hand_vel_data_train = hand_vel_data[train_list]

neural_data_valid = neural_data[valid_list]
spike_data_valid = spike_data[valid_list]
hand_vel_data_valid = hand_vel_data[valid_list]

X_train = neural_data_train
X_train_spike = spike_data_train
y_train = hand_vel_data_train

X_valid = neural_data_valid
X_valid_spike = spike_data_valid
y_valid = hand_vel_data_valid

print("Saving the dataset ...")
if fr_scale > 1:
    file_name = "/spikes_{}_duplicate_{}_{}x.npz".format("MAZE", duplicate_num, fr_scale)
else:
    file_name = "/spikes_{}_duplicate_{}.npz".format("MAZE", duplicate_num)
np.savez(save_dir + file_name, X_train = X_train, X_train_spike = X_train_spike, y_train = y_train,
                               X_valid = X_valid, X_valid_spike = X_valid_spike, y_valid = y_valid)
