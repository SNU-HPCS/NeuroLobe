import sys, os
import math
import numpy as np
from nlb_tools.nwb_interface import NWBDataset
import pandas as pd
import subprocess
import math
from pathlib import Path
import configparser

properties = configparser.ConfigParser()
properties.read('gen.params')

save_dir = properties.get('path', 'save_dir')
data_dir = properties.get('path', 'data_dir')

mode = properties.get('dataset', 'mode')
duration = properties.getint('dataset', 'duration')
fr_scale = properties.getint('dataset', 'fr_scale')

Path(save_dir).mkdir(parents=True, exist_ok=True)
Path(data_dir).mkdir(parents=True, exist_ok=True)

target_duration = duration * fr_scale
sampling_rate = 1000  # for the dandi MAZE dataset

print("Downloading the dataset ...")

# Load dataset
if mode == "BUMP":
    filepath = "./dandi_data/000127/sub-Han"
elif mode == "MAZE":
    filepath = "./dandi_data/000128/sub-Jenkins"
elif mode == "RTT":
    filepath = "./dandi_data/000129/sub-Indy"
else:
    raise ValueError("Unsupported Dataset")
if not os.path.exists(filepath):
    if mode == "BUMP":
        proc = subprocess.Popen(["dandi download https://gui.dandiarchive.org/#/dandiset/000127"], close_fds=True, shell=True, cwd=".")
        proc.communicate()
        proc.wait()
        proc = subprocess.Popen(["mv 000127 ./dandi_data/."], close_fds=True, shell=True, cwd=".")
        proc.communicate()
        proc.wait()
        vel_name = "hand_vel"
    elif mode == "MAZE":
        proc = subprocess.Popen(["dandi download https://gui.dandiarchive.org/#/dandiset/000128"], close_fds=True, shell=True, cwd=".")
        proc.communicate()
        proc.wait()
        proc = subprocess.Popen(["mv 000128 ./dandi_data/."], close_fds=True, shell=True, cwd=".")
        proc.communicate()
        proc.wait()
        vel_name = "hand_vel"
    elif mode == "RTT":
        proc = subprocess.Popen(["dandi download https://gui.dandiarchive.org/#/dandiset/000129"], close_fds=True, shell=True, cwd=".")
        proc.communicate()
        proc.wait()
        proc = subprocess.Popen(["mv 000129 ./dandi_data/."], close_fds=True, shell=True, cwd=".")
        proc.communicate()
        proc.wait()
        vel_name = "finger_vel"
    else: assert(0)

print("Parsing the dataset ...")
dataset = NWBDataset(filepath, split_heldout=False)

if mode == "RTT":
    has_change = dataset.data.target_pos.fillna(-1000).diff(axis=0).any(axis=1) # filling NaNs with arbitrary scalar to treat as one block
    # Find if target pos change corresponds to NaN-padded gap between files
    change_nan = dataset.data[has_change].isna().any(axis=1)
    # Drop trials containing the gap and immediately before and after, as those trials may be cut short
    drop_trial = (change_nan | change_nan.shift(1, fill_value=True) | change_nan.shift(-1, fill_value=True))[:-1]
    # Add start and end times to trial info
    change_times = dataset.data.index[has_change]
    start_times = change_times[:-1][~drop_trial]
    end_times = change_times[1:][~drop_trial]

    # Get target position per trial
    target_pos = dataset.data.target_pos.loc[start_times].to_numpy().tolist()
    # Compute reach distance and angle
    reach_dist = dataset.data.target_pos.loc[end_times - pd.Timedelta(1, 'ms')].to_numpy() - dataset.data.target_pos.loc[start_times - pd.Timedelta(1, 'ms')].to_numpy()
    reach_angle = np.arctan2(reach_dist[:, 1], reach_dist[:, 0]) / np.pi * 180
    # Create trial info
    dataset.trial_info = pd.DataFrame({
        'trial_id': np.arange(len(start_times)),
        'start_time': start_times,
        'end_time': end_times,
        'target_pos': target_pos,
        'reach_dist_x': reach_dist[:, 0],
        'reach_dist_y': reach_dist[:, 1],
        'reach_angle': reach_angle,
    })

if mode == "BUMP" or mode == "MAZE":
    trial_data = dataset.make_trial_data(align_field='move_onset_time')
elif mode == "RTT":
    trial_data = dataset.make_trial_data()

grouped_trial_data = list(trial_data.groupby('trial_id', sort=False))

width = 1000
sliding_window = []
sliding_window_rate = []
for trial_idx in range(len(grouped_trial_data)):
# for trial_idx in range(10):
    spike_dat = grouped_trial_data[trial_idx][1]['spikes'].to_numpy()
    interval = spike_dat.shape[0]
    for i in range(0, interval, width):
        if i + width <= interval:
            parsed_spike = spike_dat[i:i+width]

            nonzero_dat = np.nonzero(parsed_spike)
            num_spikes = np.count_nonzero(parsed_spike)
            num_total = parsed_spike.shape[0] * parsed_spike.shape[1]
            rate = float(num_spikes/num_total) * 1000
            sliding_window.append(parsed_spike)
            sliding_window_rate.append(rate)

sliding_window = np.array(sliding_window)
sliding_window_rate = np.array(sliding_window_rate)

# print(len(grouped_trial_data))
# print(sliding_window.shape)
# print(sliding_window_rate)

n = target_duration // 1000
print("Extracting", n, "trials ...")

max_idx = np.argpartition(sliding_window_rate,-n)[-n:]
min_idx = np.argpartition(sliding_window_rate,n)[:n]
mid_idx = np.argpartition(abs(sliding_window_rate - np.median(sliding_window_rate)),n)[:n]
# print(max_idx)
# print(min_idx)
# print(mid_idx)

input_list = {}
if max_idx.any():
    input_list['max'] = np.concatenate(sliding_window[max_idx])
if min_idx.any():
    input_list['min'] = np.concatenate(sliding_window[min_idx])
if mid_idx.any():
    input_list['mid'] = np.concatenate(sliding_window[mid_idx])

if fr_scale > 1:
    print("Scaling firing rate by {}x".format(fr_scale))
    for extract_mode in input_list.keys():
        input_list[extract_mode] = input_list[extract_mode].reshape(fr_scale, -1, input_list[extract_mode].shape[1]).sum(axis=0)

for extract_mode in input_list.keys():
    print("{}: {:.2f}".format(extract_mode, np.sum(input_list[extract_mode], axis=(0,1)) / input_list[extract_mode].shape[0] / input_list[extract_mode].shape[1] * sampling_rate))

print("Saving the dataset ...")
for extract_mode in input_list.keys():
    if fr_scale > 1:
        file_name = "/spikes_{}_{}_{}x".format(mode, extract_mode, fr_scale)
    else:
        file_name = "/spikes_{}_{}".format(mode, extract_mode)
    np.savez(save_dir + file_name, input = input_list[extract_mode])
