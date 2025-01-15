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

Path(save_dir).mkdir(parents=True, exist_ok=True)
Path(data_dir).mkdir(parents=True, exist_ok=True)

target_duration = duration

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


spikes_list = []
hand_vel_list = []
template_list = {}
input_list = None


if mode == 'BUMP':
    # 16 templates (target, result)
    for trial_idx in range(len(grouped_trial_data)):
        trial_info = dataset.trial_info.split[trial_idx]
        spike_dat = grouped_trial_data[trial_idx][1]['spikes'].to_numpy()
        
        if trial_info != 'test':
            target = dataset.trial_info['target_dir'][trial_idx]
            result = dataset.trial_info['result'][trial_idx]
            if not math.isnan(target) and result != 'I':
                target = int(target)
                if (target,result) in template_list:
                    template_list[(target,result)] = np.append(template_list[(target,result)], spike_dat, axis = 0)
                else:
                    template_list[(target,result)] = spike_dat

elif mode == 'MAZE':
    # 36 templates (maze id)
    for trial_idx in range(len(grouped_trial_data)):
        trial_info = dataset.trial_info.split[trial_idx]
        spike_dat = grouped_trial_data[trial_idx][1]['spikes'].to_numpy()
        
        if trial_info != 'test':
            maze_id = dataset.trial_info['maze_id'][trial_idx]
            if not math.isnan(maze_id):
                maze_id = int(maze_id)
                if maze_id in template_list:
                    template_list[maze_id] = np.append(template_list[maze_id], spike_dat, axis = 0)
                else:
                    template_list[maze_id] = spike_dat

elif mode == 'RTT':
    # 52 templates (target_pos)
    for trial_idx in range(len(grouped_trial_data)):
        # trial_info = dataset.trial_info.split[trial_idx]
        spike_dat = grouped_trial_data[trial_idx][1]['spikes'].to_numpy()
        target_pos = (dataset.trial_info['target_pos'][trial_idx][0], dataset.trial_info['target_pos'][trial_idx][1])
        if target_pos in template_list:
            template_list[target_pos] = np.append(template_list[target_pos], spike_dat, axis = 0)
        else:
            template_list[target_pos] = spike_dat

print("Saving the dataset ...")
file_name = "/templates_{}".format(mode)
np.savez(save_dir + file_name, template = template_list)

len(template_list.keys())


