import matplotlib.pyplot as plt
from pprint import pprint
from tqdm import tqdm

import spikeinterface


import spikeinterface as si  # import core only
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.postprocessing as spost
import spikeinterface.qualitymetrics as sqm
import spikeinterface.comparison as sc
import spikeinterface.exporters as sexp
import spikeinterface.curation as scur
import spikeinterface.widgets as sw

# import spikeinterface.full as si
import os

# import bci_processor api for params file generation
from bci_processor_api import *

# import sys for command line argument passing
import sys
import time
import numpy
import h5py

import pickle
import configparser

# define helper functions

def get_adj_neuron_ids(peaked_elec_id, electrode_positions, sim_radius):
    adj_elec_ids = []
    local_peaked_position = electrode_positions[peaked_elec_id]
    for elec in range(len(electrode_positions)):
        distance = numpy.linalg.norm(electrode_positions[elec] - local_peaked_position)
        if distance <= sim_radius:
            adj_elec_ids.append(elec)
    return adj_elec_ids

WORKERS = 0
MODE = 'inference'
if len(sys.argv) > 1:
    WORKERS = int(sys.argv[1])
    MODE = sys.argv[2]
assert isinstance(WORKERS,int), "number of workers should be integer value"

# default sim params
sim_real_time = False
sim_spatial_sparse = False
valid_multiple = False
sim_num_workers = 1

properties = configparser.ConfigParser()

properties.read('ss.params')

# general params
if len(sys.argv) < 2:
    sim_mode = properties.get('general','mode')
else:
    sim_mode = MODE
assert sim_mode in ("training","inference"), "MODE : training, and inference"

if len(sys.argv) < 2: # if num workers not specified
    sim_num_workers = properties.getint('general','num_workers')
else:
    sim_num_workers = WORKERS

dataset = properties.get('general','dataset')
data_name = properties.get('general','data_name')
result_dir = properties.get('general','result_dir')
sort_result_dir = properties.get('general','sort_result_dir')

# detection params
sim_sign_peaks = properties.getint('detection', 'sign_peaks')
sim_detect_threshold = properties.getint('detection', 'detect_threshold')

# training params
sim_sparsify = properties.getfloat('training','sparsify')
sim_spatial_whitening = properties.getboolean('training','spatial_whitening')
sim_radius = properties.getfloat('training','radius') # NOTE : this parameter is also used in inference
sim_radius_en = properties.getboolean('training','radius_en')

# inference params
sim_real_time = properties.getboolean('inference', 'real_time')
sim_spatial_sparse = properties.getboolean('inference','spatial_sparse')
valid_multiple = properties.getboolean('inference','valid_multiple')
sim_duration = properties.getint('inference','duration')

# result params
sim_snr_threshold = properties.getint('result', 'snr_threshold')

if sim_mode == 'inference':
    sim_num_workers = 1

# check user-provided simulation params
print("-------------------------------")
print("Simulation Parameters")
print(f"realtime: {sim_real_time}")
print(f"spatial sparse: {sim_spatial_sparse}")
print(f"valid once: {valid_multiple}")
print(f"sparsify: {sim_sparsify}")
print(f"spatial whitening: {sim_spatial_whitening}")
print(f"cpus: {sim_num_workers}")
print(f"simluation duration: {sim_duration} (ms)")
print(f"sign_peaks: {sim_sign_peaks}")

if not os.path.exists(sort_result_dir):
    os.makedirs(sort_result_dir)
sorter_result_path = sort_result_dir + data_name + "_" + str(sim_sparsify)

global_job_kwargs = dict(n_jobs=1, chunk_duration="1s")
si.set_global_job_kwargs(**global_job_kwargs)

# extract file extension from string 'dataset'
filename, file_extension = os.path.splitext(dataset)

filename = filename.split('/')[-1]
print("filename:", filename)
print("-------------------------------")

# declare
recording = 0
sorting_true = 0

if file_extension == '.h5':
    print("MEArec dataset")
    # MEArec dataset - h5
    recording, sorting_true = se.read_mearec(dataset)
elif file_extension == '.nwb':
    print("spikeinterface dataset")
    # Spikeinterface dataset - nwb
    sorting_true = se.NwbSortingExtractor(dataset)
    recording = se.NwbRecordingExtractor(dataset)
else:
    print(f"not implemented for file extension: {file_extension}")
    print("this code does not support spikeforest dataset yet")
    exit()
temp_path = sorter_result_path + '/sorter_output/recording/recording.templates.hdf5'

recording_preprocessed = recording

print("recording", recording)
print("sorting_true", sorting_true)

# Get necessary simulation parameters here
# from recording(MEArecRecordingExtractor)
freq = recording.get_sampling_frequency()
DT = 1000 / freq
DURATION = recording.get_total_duration() # s
chan_ids = recording.get_channel_ids() # neuron_num
chan_loc = recording.get_channel_locations()
N_e = len(chan_ids)

dataset_result_path = result_dir + "SS_" + str(len(chan_ids)) + "_" + str(sim_sparsify)
print("\ndataset result path:", dataset_result_path)


print("-------------------------------")
print("Recording Parameters")
print("frequency:", freq)
print("duration:", DURATION, "(s)")
# print("channel ids: ", chan_ids)
print("-------------------------------")
assert(DURATION * 1000 >= sim_duration)

duration = int(sim_duration/ DT) 

# recording data for spykingcircus ready, init simulation
# 1. INIT SIMULATION
dt = DT
temp_shape = h5py.File(temp_path, 'r', libver='earliest').get('temp_shape')[:] # template shape : N_e x N_t x N_tm
nb_templates = temp_shape[-1] // 2 # only number of primary templates
N_t = temp_shape[1]
init_simulation(duration, dt, N_e, nb_templates, N_t, dataset_result_path)

# 2. CREATE TEMPLATES and relevant params

temp_x = h5py.File(temp_path, 'r', libver='earliest').get('temp_x')[:].ravel()
temp_y = h5py.File(temp_path, 'r', libver='earliest').get('temp_y')[:].ravel()
temp_data = h5py.File(temp_path, 'r', libver='earliest').get('temp_data')[:].ravel()
norm_templates = h5py.File(temp_path, 'r', libver='earliest').get('norms')[:]
amp_limits = h5py.File(temp_path, 'r', libver='earliest').get('limits')[:]

# densify the template
N_tm = temp_shape[-1] # number of primary templates + secondary templates
sparse_mat = numpy.zeros((N_e*N_t, N_tm), dtype=numpy.float32)
sparse_mat[temp_x, temp_y] = temp_data

# normalize
#norm_templates = load_data(params, 'norm-templates')
for idx in range(sparse_mat.shape[1]):
    sparse_mat[:, idx] /= norm_templates[idx]

# transpose
templates = sparse_mat.T # N_tm x (N_e*N_t)

threshold_path = sorter_result_path + '/sorter_output/recording/recording.thresholds.npy'
thresholds = np.load(threshold_path)

#create_initial_states(norm_templates, amp_limits, thresholds, nb_templates, N_e*N_t)

# 3. create external recording data
file_name = sorter_result_path + '/sorter_output/recording.npy'
whitened_file_name = sorter_result_path + '/sorter_output/whitened_recording.npy'
if sim_spatial_whitening:
    # print("update & run spyking circus to get whitened_recording.npy")
    elec_data = numpy.load(whitened_file_name, mmap_mode='r') # elec data shape 320000 * 32
else:
    elec_data = numpy.load(file_name, mmap_mode='r') # elec data shape 320000 * 32

template_shift = N_t // 2

external_spikes = []
recording_time = duration
spike_boundary = (template_shift, recording_time - template_shift)

# print("recording_time", recording_time)
# print("construct adj elec dict")

electrode_positions = numpy.load(sorter_result_path + '/sorter_output/electrode_positions.npy')

adj_elec_dict = {}
for peaked_elec in range(len(electrode_positions)):
    local_peaked_position = electrode_positions[peaked_elec]
    adj_elec_ids = []
    for elec in range(len(electrode_positions)):
        distance = numpy.linalg.norm(electrode_positions[elec] - local_peaked_position)
        if distance <= sim_radius:
            adj_elec_ids.append(elec)
    adj_elec_dict[peaked_elec] = adj_elec_ids


total_count = 0
spike_count = 0
# print("generating external spikes data")
# predefined valid timestep list
valid_timestep_list = []

# The neurons to transfer and their boundary
neuron_info = [-1 for _ in range(N_e)]
for timestep in tqdm(range(recording_time)):
    # if timestep % 10000 == 0:
    #     print(f"{timestep}/{recording_time}")
    if timestep < spike_boundary[0]:
        continue
    if timestep >= spike_boundary[1]:
        break
    to_send_neuron_ids = []
    for neuron_id in range(N_e):
        curr_dat = - elec_data[timestep,neuron_id]
        if curr_dat > thresholds[neuron_id]:
            prev_dat = - elec_data[timestep - 1,neuron_id]
            next_dat = - elec_data[timestep + 1,neuron_id]
            if curr_dat > prev_dat and curr_dat > next_dat:
                adj_neuron_ids = adj_elec_dict[neuron_id]
                # concate adj neuron ids
                to_send_neuron_ids += adj_neuron_ids
    # union of adj_neuron_ids at this timestep
    to_send_neuron_ids = list(set(to_send_neuron_ids))

    # update the neuron info
    for neuron_id in to_send_neuron_ids:
        neuron_info[neuron_id] = timestep + 2 * template_shift
    
    for neuron_id in range(N_e):
        if neuron_info[neuron_id] >= timestep:
            sending_timestep = timestep - template_shift
            data = elec_data[sending_timestep][neuron_id]
            total_count += 1
            if abs(data) > 1.125:
                spike_count += 1
                external_spikes.append((sending_timestep, neuron_id, elec_data[sending_timestep][neuron_id]))

    #for neuron_id in to_send_neuron_ids:
    #    for sending_timestep in range(timestep - template_shift, timestep + template_shift + 1):
    #        total_count += 1
    #        if -1 < elec_data[sending_timestep][neuron_id] < 1:
    #            non_spike_count += 1
    #            external_spikes.append((sending_timestep, neuron_id, elec_data[sending_timestep][neuron_id]))

# print("start sorting external spikes")

#external_spikes = list(set(external_spikes))
#external_spikes.sort(key=lambda x: x[0])
#
## print("done sorting external spikes")
#
create_external_stimulus(num_external  = N_e,
                  external_stim = external_spikes)

# 4. CREATE CONNECTIONS
create_connections(temp_shape, templates)

# 5. END SIMULATION
end_simulation()
