import matplotlib.pyplot as plt
from pprint import pprint

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

# run spyking circus

sim_with_output = True
if sim_mode == "training":
    sim_with_output = False

if not os.path.exists(sorter_result_path):
    os.makedirs(sorter_result_path)

if sim_with_output:
    # We should copy from the result to the spyking circus output
    #os.system('cp -r ' + sorter_result_path + '/* ./spykingcircus_output/. ')
    os.system('rsync -av --progress ' + sorter_result_path + '/* ./spykingcircus_output/. --exclude waveforms')
    

start = time.time()

if sim_mode == "training":
    sorting_SC = ss.run_sorter(sorter_name="spykingcircus", recording=recording_preprocessed, verbose=False,
            with_output=sim_with_output, num_workers=sim_num_workers, realtime=sim_real_time, spatial=sim_spatial_sparse,
            valid_multiple=valid_multiple, mode=sim_mode, sparsify=sim_sparsify, spatial_whitening=sim_spatial_whitening, duration=duration,
            radius=sim_radius, detect_sign = sim_sign_peaks, detect_threshold = sim_detect_threshold, radius_en = sim_radius_en,
            output_folder = sorter_result_path)
else:
    sorting_SC = ss.run_sorter(sorter_name="spykingcircus", recording=recording_preprocessed, verbose=False,
            with_output=sim_with_output, num_workers=sim_num_workers, realtime=sim_real_time, spatial=sim_spatial_sparse,
            valid_multiple=valid_multiple, mode=sim_mode, sparsify=sim_sparsify, spatial_whitening=sim_spatial_whitening, duration=duration,
            radius=sim_radius, detect_sign = sim_sign_peaks, detect_threshold = sim_detect_threshold, radius_en = sim_radius_en)

end = time.time()
sim_time = end - start
if sim_mode == "inference":
    print(f"INFERENCE DONE : {sim_time}")
elif sim_mode == "training":
    print(f"TRAINING DONE : {sim_time}")
    # print(sorting_SC)

if not sim_mode == "training":
    os.system('cp -r ./spykingcircus_output/* ' + sorter_result_path)

if sim_mode == "training":
    exit(0)

print("\n Sort Results")
# print & save result
recording.annotate(is_filtered=True)

we = si.extract_waveforms(recording = recording, sorting = sorting_true, folder = sorter_result_path + '/waveforms', overwrite = True)
snrs = sqm.compute_snrs(we)
# print(snrs)

comp_gt = sc.compare_sorter_to_ground_truth(gt_sorting=sorting_true, tested_sorting=sorting_SC)
perf = comp_gt.get_performance()
# print(perf)

accuracy = perf.accuracy.to_list()

accuracy_sum = 0
num_valid_gt_units = 0
for i, snr in enumerate(snrs.values()):
    if snr > sim_snr_threshold:
        # add to average
        accuracy_sum += accuracy[i]
        num_valid_gt_units += 1

accuracy_summary_str = f"\nAverage accuracy above SNR threshold {sim_snr_threshold} : {accuracy_sum / num_valid_gt_units}"
print(accuracy_summary_str)

csv_file_name = f"{sorter_result_path}/performance.csv"
perf.to_csv(csv_file_name)
perf_file = open(csv_file_name, 'a')
perf_file.write('\n' + accuracy_summary_str)
perf_file.close()
print("Saved accuracy result into csv file")
