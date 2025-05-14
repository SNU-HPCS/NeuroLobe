import sys
import numpy as np
import time
import scipy

from bci_processor_api import *

from scipy.ndimage import gaussian_filter1d
from neo.core import AnalogSignal
import quantities as pq
from elephant import spike_train_generation
import configparser
import networkx as nx

np.random.seed(0)

def load_data(dataset):
    data = np.load(dataset, allow_pickle=True)
    total_spike_mat = np.transpose(data['input'])

    return total_spike_mat

def pairwise_correlation(spike_mat, cell_num, duration, window, corr_period, conn_list):
        # binning
    spike_mat = np.sum(spike_mat, axis = 2)

    corr = [[[0 for _ in range(window * 2 + 1)] for _ in range(cell_num)] for _ in range(cell_num)] # Calculate through HW-like method
    debug_corr = [[[0 for _ in range(window * 2 + 1)] for _ in range(cell_num)] for _ in range(cell_num)] # Direct calculation for debugging

    spike_total = spike_mat.shape[1]

    partial_corr = [[[0 for _ in range(window + 1)] for _ in range(cell_num)] for _ in range(cell_num)] # Calculate through HW-like method
    num_spikes = [0 for _ in range(cell_num)] # Number of spikes so far

    pcc = [[] for _ in range(cell_num)]

    print("\nHW-like Method")
    # Calculate pairwise correlation with partial correlograms
    for t in range(spike_total):
        for reference in range(cell_num):
            ref = spike_mat[reference]
            spiked = ref[t]
            num_spikes[reference] += spiked ** 2
            # spike at reference neuron
            if spiked > 0:
                for target in range(cell_num):
                    if target != reference and conn_list[reference][target]:
                        tar = spike_mat[target]

                        for i in range(max(0, t - window), t + 1):
                            # spike history at target neuron
                            if tar[i]:
                                tar_idx = t - i
                                partial_corr[target][reference][tar_idx] += tar[i] * spiked


        # print result every corr_period
        if t % corr_period == 0:

            # partial correlation concatenation
            for reference in range(cell_num):
                for target in range(cell_num):
                    past = list(reversed(partial_corr[reference][target]))
                    future = partial_corr[target][reference]

                    for i in range(len(past)):
                        corr[target][reference][i] = past[i]
                    for i in range(len(future) - 1):
                        corr[target][reference][i + len(past)] = future[i + 1]

            # calculate correlation from correlogram
            for target in range(cell_num):
                pcc[target].append({})
                for reference in range(cell_num):
                    if target != reference and conn_list[target][reference]:
                        reference_spike = num_spikes[reference]
                        target_spike = num_spikes[target]
                        p = np.nan_to_num(partial_corr[target][reference] / np.sqrt(reference_spike * target_spike)).tolist()
                        if sum(p):
                            pcc[target][-1][reference] = p


    debug_file = open(result_path + '/multicore_spike_out.dat', 'w')
    for i in range(cell_num):
        debug_file.write(str(i) + ": " + str(pcc[i]) + "\n")
    debug_file.close()

if __name__ == '__main__':
    # Load simulation parameters
    properties = configparser.ConfigParser()
    properties.read('pc.params')

    data_path = properties.get('general','data_path')
    data_mode = properties.get('general','data_mode')
    result_dir = properties.get('general','result_dir')

    sim_duration = properties.getint('general','duration')
    target_cell_num = properties.getint('pairwise_correlation','target_cell_num')
    sampling_rate = properties.getint('general','sampling_rate')

    bin_width = properties.getint('pairwise_correlation','bin_width') * sampling_rate // 1000
    window = properties.getint('pairwise_correlation','window')
    corr_period = properties.getint('pairwise_correlation','corr_period')

    epsilon = properties.getfloat('network_config','epsilon')
    beta = properties.getfloat('network_config','beta')

    # Load spike data
    total_spike_mat = load_data(data_path)

    cell_num, duration = total_spike_mat.shape
    duration = min(sim_duration, duration)
    total_spike_mat = total_spike_mat[:, :duration]
    duplicate_num = target_cell_num // cell_num + 1

    result_path = result_dir + "PC_" + str(target_cell_num) + "_" + str(data_mode) + "_" + str(bin_width) + "_" + str(corr_period)

    print("=====================================")
    print("duration:", duration)
    print("cell num:", cell_num)
    print("duplicate num:", duplicate_num)
    print("target cell num:", target_cell_num)
    print("bin width:", bin_width, "(ms)")
    print("window:", window, "(bins)")
    print("sampling rate:", sampling_rate)
    print("epsilon:", epsilon)
    print("beta:", beta)
    print("result path:", result_path)
    print("=====================================")

    # Scale data by duplicate_num
    new_total_spike_mat = []

    if duplicate_num == 1:
        new_total_spike_mat = total_spike_mat
    else:
        # Scale spike input data
        # Retrieve average firing rate using gaussian filter
        # Averate the filter using
        gaussian_spike = np.zeros(total_spike_mat.shape)

        # Generate new shape
        new_spike_shape = (total_spike_mat.shape[0] * duplicate_num, total_spike_mat.shape[1])
        generated_spikes = np.zeros(new_spike_shape)

        for cell in range(cell_num):
            gaussian_spike[cell, :] = gaussian_filter1d(total_spike_mat[cell, :].astype('float32'), 100000 / sampling_rate)
            # To calculate the instant rate
            gaussian_spike[cell, :] = gaussian_spike[cell, :] * sampling_rate
            # Retrieve analog signal
            analog_spike = AnalogSignal(gaussian_spike[cell, :], units='Hz', sampling_rate=sampling_rate*pq.Hz, refractory=1000/sampling_rate*pq.ms)
            spike_generator = spike_train_generation.NonStationaryPoissonProcess(analog_spike)

            for dup_id in range(duplicate_num):
                spike_time = spike_generator._generate_spiketrain_as_array()
                # Generate random spike times
                spike_time = spike_time * sampling_rate
                spike_time = np.array(spike_time, dtype=int)
                spike_list = np.zeros(total_spike_mat.shape[1], dtype=int)
                for spk_time in spike_time:
                    spike_list[spk_time] = 1

                generated_spikes[cell * duplicate_num + dup_id, :] = spike_list

        new_total_spike_mat = generated_spikes

    original_cell_num = cell_num
    cell_num = cell_num * duplicate_num

    print("\nduplicate done by", duplicate_num)

    if target_cell_num != -1:
        assert(target_cell_num <= cell_num)
        cell_num = target_cell_num
        new_total_spike_mat = new_total_spike_mat[:target_cell_num, :]

        print("removed cells to match target cell num")

    print("new cell num:", cell_num)
    print("firing rate: {:.2f} -> {:.2f}".format(np.sum(total_spike_mat, axis=(0,1)) / original_cell_num / duration * sampling_rate, \
            np.sum(new_total_spike_mat, axis=(0,1)) / cell_num / duration * sampling_rate))

    # Make random connection between neurons
    conn_list = []
    np.random.seed(0)
    conn = nx.watts_strogatz_graph(cell_num, int(cell_num * epsilon), beta, seed=0).edges()
    for src, dst in conn:
        if src != dst:
            conn_list.append([src, dst])
            conn_list.append([dst, src])
    conn_list = sorted(conn_list)

    # perform pairwise correlation in SW
    new_total_spike_mat = np.array(new_total_spike_mat, dtype = np.uint32)
    M_t = int(new_total_spike_mat.shape[-1] / bin_width)

    spike_mat = new_total_spike_mat[:, :M_t * bin_width].reshape(cell_num, -1, bin_width)

    # Make random connection between neurons
    conn_mat = np.zeros((cell_num, cell_num), dtype=np.uint32)
    for src, dst in conn:
        conn_mat[src,dst] = 1
        conn_mat[dst,src] = 1

    conn_density = np.count_nonzero(conn_mat)/(cell_num*cell_num)
    print("Conn density: ", conn_density)

    print("Parsing benchmark...")
    # 1. INIT SIMULATION
    init_simulation(duration / bin_width, 1, cell_num, window, corr_period, result_path)

    # 2. CREATE NEURONS
    # print(cell_num)

    # 3. CREATE EXTERNAL STIMULUS
    spike_data = []
    for neuron_id in range(cell_num):
        for timestep in range(duration):
            if new_total_spike_mat[neuron_id, timestep]:
                spike_data.append((timestep // bin_width, neuron_id, 1)) # spiketime / spiked

    external_stim = sorted(spike_data) # sort spike data according to the spiked timestep
    create_external_stimulus(cell_num, external_stim)

    # 4. CREATE CONNECTIONS
    # Make random connection between neurons
    create_connections(cell_num, conn_list)

    # 5. END SIMULATION
    end_simulation()

    pairwise_correlation(spike_mat, cell_num, duration / bin_width, window, corr_period, conn_mat)
