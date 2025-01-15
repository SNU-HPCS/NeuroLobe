import sys
import numpy as np
import time

from bci_processor_api import *

from scipy.ndimage import gaussian_filter1d
from neo.core import AnalogSignal
import quantities as pq
from elephant import spike_train_generation
import configparser

np.random.seed(0)

def load_data(dataset):
    data = np.load(dataset, allow_pickle=True)
    total_spike_mat = np.transpose(data['input'])

    return total_spike_mat

def load_templates(dataset, template_window):
    data = np.load(dataset, allow_pickle=True)
    
    temp_data = data['template'][()]

    temp_mat_list = []
    temp_neuron_list = []
    for temp_id in temp_data:
        template = np.transpose(temp_data[temp_id]).astype(int)
        assert(template.shape[1] > template_window)
        temp_mat_list.append(template[:, :template_window])
        temp_neuron_list.append(np.arange(template.shape[0], dtype=np.uint32))

    # print([temp.shape for temp in temp_mat_list])
    return temp_mat_list, temp_neuron_list


def template_matching_with_psums(spike_mat, temp_mat_list, bin_width, temp_to_use):
    N = temp_mat_list.shape[1]
    M_t = spike_mat.shape[1]

    # list of all corr values obtained by sliding the time window
    corr = [[] for _ in temp_to_use]

    C1 = [0  for _ in temp_to_use]
    C2 = [0  for _ in temp_to_use]
    C3 = [0  for _ in temp_to_use]

    binned_spike_mat = np.sum(spike_mat, axis=2)
    # print(binned_spike_mat.shape)
    # print(binned_spike_mat.sum(axis=1))

    for temp_idx in range(len(temp_to_use)):
        temp_mat = temp_mat_list[temp_idx]

        binned_temp_mat = np.sum(temp_mat, axis=2)
        # print(binned_temp_mat.shape)

        # print(np.count_nonzero(binned_temp_mat), binned_temp_mat.shape)

        n_t = binned_temp_mat.shape[-1]

        # Compute Constants
        c1 = n_t * N
        c2 = 0
        c3 = 0
        for m in range(n_t):
            for n in range (N):
                c2 += binned_temp_mat[n, m]
                c3 += binned_temp_mat[n, m] ** 2
        c3 = c1 * c3 - c2 ** 2

        C1[temp_idx] = c1
        C2[temp_idx] = c2
        C3[temp_idx] = c3

        this_corr = []

        # Initialize R2, R3, S2, S3
        S1 = 0
        S2 = 0
        S3 = 0
        R2 = [0 for _ in range(n_t)]
        R3 = [0 for _ in range(n_t)]
        for t in range(n_t - 1):
            P2 = 0
            P3 = 0
            for n in range(N):
                w = binned_spike_mat[n][t]
                P2 += w
                P3 += w ** 2
            S2 += P2
            R2[t] = P2
            S3 += P3
            R3[t] = P3

        # Compute Summations
        for t in range(n_t - 1, binned_spike_mat.shape[-1]):
            target_spike_mat = binned_spike_mat[:,t-n_t+1:t+1]
            # print(target_spike_mat.shape)
            # print(target_spike_mat.sum(axis=1))

            P2 = 0
            P3 = 0
            S1 = 0
            for n in range(N):
                S1 += np.sum(binned_temp_mat[n] * target_spike_mat[n])
                w = binned_spike_mat[n][t]
                P2 += w
                P3 += w ** 2
            S2 = S2 + P2 - R2[t % n_t]
            R2[t % n_t] = P2
            S3 = S3 + P3 - R3[t % n_t]
            R3[t % n_t] = P3

            x = ((c1 * S1 - c2 * S2) ** 2) / (c3 * (c1 * S3 - S2 ** 2))
            #print("S1", S1, "S2", S2,"S3", S3, x)
            this_corr.append(x)

        this_corr = np.array(this_corr)
        corr[temp_idx] = this_corr

    return corr, C1, C2, C3


if __name__ == '__main__':
    # Load simulation parameters
    properties = configparser.ConfigParser()
    properties.read('tm.params')

    spike_data = properties.get('general','spike_data')
    template_data = properties.get('general','template_data')
    data_mode = properties.get('general','data_mode')
    result_dir = properties.get('general','result_dir')

    sim_duration = properties.getint('general','duration')
    sampling_rate = properties.getint('general','sampling_rate')

    template_window = properties.getint('template_generation','template_window') * sampling_rate

    target_cell_num = properties.getint('template_matching','target_cell_num')
    # duplicate_num = properties.getint('template_matching','duplicate_num')
    bin_width = properties.getint('template_matching','bin_width') * sampling_rate // 1000
    temp_to_use = properties.get('template_matching','temp_to_use')

    # test_to_use = properties.get('general','test_to_use')

    # Load spike & template data
    total_spike_mat = load_data(spike_data)
    temp_mat_list, temp_neuron_list = load_templates(template_data, template_window)

    cell_num, duration = total_spike_mat.shape
    duration = min(sim_duration, duration)
    total_spike_mat = total_spike_mat[:, :duration]
    duplicate_num = (target_cell_num - 1) // cell_num + 1

    if temp_to_use == 'all':
        temp_to_use = range(len(temp_mat_list))
    else:
        temp_to_use = eval(temp_to_use)

    result_path = result_dir + "TM_" + str(target_cell_num) + "_" + str(data_mode) + "_" + str(bin_width)

    print("=====================================")
    print("duration:", duration)
    print("cell num:", cell_num)
    print("template num:", len(temp_mat_list))
    print("duplicate num:", duplicate_num)
    print("target cell num:", target_cell_num)
    print("bin width:", bin_width, "(ms)")
    print("template window:", template_window, "(s)")
    print("sampling rate:", sampling_rate)
    print("temp to use:", temp_to_use)
    print("result path:", result_path)
    print("=====================================")


    # Scale data by duplicate_num
    new_temp_mat_list = []
    new_total_spike_mat = []

    if duplicate_num == 1:
        new_total_spike_mat = total_spike_mat
        new_temp_mat_list = temp_mat_list
    else:
        # Scale spike input data
        # Retrieve average firing rate using gaussian filter
        # Average the filter using
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

        # Scale template data
        for i in range(len(temp_mat_list)):
            total_temp_mat = temp_mat_list[i]
            temp_neuron_list[i] = np.array([], dtype=int)

            # Retrieve average firing rate using gaussian filter
            # Averate the filter using
            gaussian_temp = np.zeros(total_temp_mat.shape)
            # Generate new shape
            new_temp_shape = (total_temp_mat.shape[0] * duplicate_num, total_temp_mat.shape[1])
            generated_temps = np.zeros(new_temp_shape)

            for cell in range(cell_num):
                gaussian_temp[cell, :] = gaussian_filter1d(total_temp_mat[cell, :].astype('float32'), 100000 / sampling_rate)
                # To calculate the instant rate
                gaussian_temp[cell, :] = gaussian_temp[cell, :] * sampling_rate
                # Retrieve analog signal
                analog_temp = AnalogSignal(gaussian_temp[cell, :], units='Hz', sampling_rate=sampling_rate*pq.Hz, refractory=1000/sampling_rate*pq.ms)
                temp_generator = spike_train_generation.NonStationaryPoissonProcess(analog_temp)

                for dup_id in range(duplicate_num):
                    temp_time = temp_generator._generate_spiketrain_as_array()
                    # Generate random spike times
                    temp_time = temp_time * sampling_rate
                    temp_time = np.array(temp_time, dtype=int)
                    temp_list = np.zeros(total_temp_mat.shape[1], dtype=int)
                    for spk_time in temp_time:
                        temp_list[spk_time] = 1
                    
                    generated_temps[cell * duplicate_num + dup_id, :] = temp_list
                    
                    # Update template-neuron connection
                    # if np.sum(temp_list):
                    #     # print(np.sum(temp_list))
                    #     temp_neuron_list[i] = np.append(temp_neuron_list[i], cell * duplicate_num + dup_id)


            new_temp_mat_list.append(generated_temps)

    original_cell_num = cell_num
    cell_num = cell_num * duplicate_num
    
    print("\nduplicate done by", duplicate_num)
    
    if target_cell_num != -1:
        assert(target_cell_num <= cell_num)
        cell_num = target_cell_num
        new_total_spike_mat = new_total_spike_mat[:target_cell_num, :]
        
        for i in range(len(new_temp_mat_list)):
            new_temp_mat_list[i] = new_temp_mat_list[i][:target_cell_num, :]
        
        for i in range(len(temp_neuron_list)):
            # temp_neuron_list[i] = temp_neuron_list[i][temp_neuron_list[i] < target_cell_num]
            temp_neuron_list[i] = np.arange(target_cell_num, dtype=np.uint32)

        print("removed cells to match target cell num")

    print("new cell num:", cell_num)
    print("firing rate: {:.2f} -> {:.2f}".format(np.sum(total_spike_mat, axis=(0,1)) / original_cell_num / duration * sampling_rate, \
            np.sum(new_total_spike_mat, axis=(0,1)) / cell_num / duration * sampling_rate))

    # Check if all the elements contain the same list
    prev_neu = np.array([-1])
    for temp_id in temp_to_use:
        if (prev_neu != -1).all():
            assert(list(prev_neu) == list(temp_neuron_list[temp_id]))
        prev_neu = temp_neuron_list[temp_id]

    # bin and duplicate the data for preprocessing
    N = new_temp_mat_list[0].shape[0]
    N_t = int(new_temp_mat_list[0].shape[-1] / bin_width)
    M_t = int(new_total_spike_mat.shape[-1] / bin_width)
    N_tm = len(temp_to_use)

    temp_mat_list = np.zeros((len(temp_to_use), N, N_t, bin_width), dtype=np.int32)

    for temp_idx in range(len(temp_to_use)):
        temp_mat = new_temp_mat_list[temp_to_use[temp_idx]]
        temp_neu = temp_neuron_list[temp_to_use[temp_idx]] 

        temp_mat_list[temp_idx] = temp_mat[temp_neu, :N_t * bin_width].reshape(N, -1, bin_width)
    
    spike_mat_list = new_total_spike_mat[:, :M_t * bin_width].reshape(N, -1, bin_width)

    corr, c1, c2, c3 = \
        template_matching_with_psums(spike_mat_list, temp_mat_list, bin_width, temp_to_use)


    # 1. INIT SIMULATION
    init_simulation(duration // bin_width, 1, cell_num, N_tm, N_t, result_path)

    # 2. CREATE TEMPLATES / NEURONS
    temp_consts = [[] for _ in range(N_tm)]
    for tm in range(N_tm):
        temp_consts[tm].append(c1[tm])
        temp_consts[tm].append(c2[tm])
        temp_consts[tm].append(c3[tm])
    create_units(temp_consts)

    # 3. CREATE EXTERNAL STIMULUS
    spike_data = []
    for neuron_id in range(cell_num):
        for timestep in range(duration):
            if new_total_spike_mat[neuron_id, timestep]:
                spike_data.append((timestep // bin_width, neuron_id, 1)) # spiketime / spiked

    external_stim = sorted(spike_data) # sort spike data according to the spiked timestep
    create_external_stimulus(cell_num, external_stim)

    # 4. CREATE CONNECTIONS
    binned_temp_mat = []
    for temp_id in temp_to_use:
        temp_mat = new_temp_mat_list[temp_id]
        binned_temp_mat.append(temp_mat[:, :temp_mat.shape[-1] - temp_mat.shape[-1] % bin_width].reshape(cell_num, -1, bin_width).sum(axis=2).tolist())
    # print(temp_neuron_list)
    # print(binned_temp_mat)
    create_connections(cell_num, N_t, temp_neuron_list, binned_temp_mat, bin_width)

    # 5. END SIMULATION
    end_simulation()


    debug_file = open(result_path + '/multicore_spike_out.dat', 'w')

    for temp_id in range(len(corr)):
        # print("\nTemplate", temp_to_use[temp_id])
        #print(corr[temp_id])
        debug_corr = np.array(corr[temp_id])
        show_corr = debug_corr[np.logical_not(np.isnan(debug_corr))]
        show_index = np.argwhere(np.logical_not(np.isnan(debug_corr)))

        # print("correlation")
        # print(show_corr)
        # print("index")
        # print(show_index.tolist())
        # print()
        result_list = [[corr, index[-1]] for corr, index in zip(show_corr, show_index)]
        debug_file.write(str(temp_id) + ": " + str(result_list) + "\n")

    debug_file.close()