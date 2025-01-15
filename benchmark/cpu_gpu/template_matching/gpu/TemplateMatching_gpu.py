import sys
import numpy as np
import cupy as cp

import time
import timeit
import scipy

from scipy.ndimage import gaussian_filter1d
from neo.core import AnalogSignal
import quantities as pq
from elephant import spike_train_generation
import configparser
from cupyx.profiler import benchmark
import argparse

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

# @profile
def template_matching_gpu_realtime(temp_mat_list, bin_width, temp_to_use, sparse, nb_chunks, chunk_size, M_t):
    print("Initiated template matching")
    elapsed = 0
    N = temp_mat_list.shape[1]
    N_t = temp_mat_list.shape[2]
    temp_size = N * N_t

    # cpu to gpu memcpy
    temp_mat_list = cp.asarray(temp_mat_list)

    binned_temp_mat_list = cp.zeros((len(temp_to_use), N, N_t), dtype=cp.int32)
    binned_temp_mat_list_sq = cp.zeros((len(temp_to_use), N, N_t), dtype=cp.int64)

    num_corr = M_t - N_t + 1
    corr = cp.zeros((len(temp_to_use), num_corr), dtype=cp.float32)

    print("Init Done")

    # Calc binning here
    binned_temp_mat_list = cp.sum(temp_mat_list, axis = 3)
    binned_temp_mat_list_sq = binned_temp_mat_list ** 2

    c1 = (N_t * N) * cp.ones(len(temp_to_use))
    c2 = cp.sum(binned_temp_mat_list[:,:,:N_t], axis = (1,2), dtype=cp.int64)
    c3 = cp.sum(binned_temp_mat_list_sq[:,:,:N_t], axis = (1,2), dtype=cp.int64)
    c3 = c1 * c3 - c2 ** 2

    binned_spike_mat_list = cp.zeros((N, N_t), dtype=cp.int32)
    binned_spike_mat_list_sq = cp.zeros((N, N_t), dtype=cp.int64)

    if sparse:
        # dimension of sparse matrix should be <= 2 
        # sparse matrix only supports bool, float32, float64, complex64 and complex128
        sparse_binned_temp_mat_list = cp.sparse.csr_matrix(binned_temp_mat_list.reshape(len(temp_to_use), temp_size).astype(cp.float32))
        sparse_binned_temp_mat_list_sq = sparse_binned_temp_mat_list.multiply(sparse_binned_temp_mat_list)

    for chunk_idx in range(nb_chunks):
        # load spike_mat
        offset = chunk_idx*chunk_size // bin_width
        new_total_spike_mat = np.load('new_total_spike_mat.npy', allow_pickle=True, mmap_mode='r')
        spike_mat = new_total_spike_mat[:, offset*bin_width:offset*bin_width+chunk_size].reshape(N, -1, bin_width)
        m_t = spike_mat.shape[1]
        spike_mat = cp.asarray(spike_mat)
        tic=timeit.default_timer()
        for t in range(m_t):
            # Dynamic binning here
            binned_spike_mat_list[:,N_t-1] = cp.sum(spike_mat[:,t,:], axis=1)
            binned_spike_mat_list_sq[:,N_t-1] = binned_spike_mat_list[:,N_t-1] ** 2

            # if sparse:
            #     sparse_binned_spike_mat_list = cp.sparse.csr_matrix(binned_spike_mat_list.astype(cp.float32))
            #     sparse_binned_spike_mat_list_sq = cp.sparse.csr_matrix(binned_spike_mat_list_sq.astype(cp.float32))

            if sparse:
                # flatten is not supported for sparse matrix
                # reshaping csr matrix returns coo matrix => todense or tocsr needed
                S1 = sparse_binned_temp_mat_list.multiply(binned_spike_mat_list.flatten()).sum(axis=1, dtype=cp.int32).flatten()
                S2 = cp.sum(binned_spike_mat_list, axis= (0, 1), dtype=cp.int64)
                S3 = cp.sum(binned_spike_mat_list_sq, axis= (0, 1), dtype=cp.int64)
            else:
                S1 = cp.sum(binned_temp_mat_list * binned_spike_mat_list, axis= (1,2), dtype=cp.int32)
                S2 = cp.sum(binned_spike_mat_list, axis= (0, 1), dtype=cp.int64)
                S3 = cp.sum(binned_spike_mat_list_sq, axis= (0, 1), dtype=cp.int64)
            x = ((c1 * S1 - c2 * S2) ** 2) / (c3 * (c1 * S3 - S2 ** 2))
            if t+offset >= N_t - 1:
                corr[:,t + offset - N_t + 1] = x

            # roll is not supported for sparse matrix
            binned_spike_mat_list = cp.roll(binned_spike_mat_list, -1, axis = 1)
            binned_spike_mat_list_sq = cp.roll(binned_spike_mat_list_sq, -1, axis = 1)

            cp.cuda.Device().synchronize()
        toc=timeit.default_timer()
        elapsed += (toc-tic)

        del new_total_spike_mat

    corr = cp.asnumpy(corr)
    c1 = cp.asnumpy(c1)
    c2 = cp.asnumpy(c2)
    c3 = cp.asnumpy(c3)
    return corr, c1, c2, c3, elapsed

def template_matching_gpu_static(spike_mat, temp_mat_list, bin_width, temp_to_use):
    print("Initiated template matching")
    N = temp_mat_list.shape[1]
    N_t = temp_mat_list.shape[2]
    M_t = spike_mat.shape[2]

    # cpu to gpu memcpy
    spike_mat = cp.asarray(spike_mat)
    temp_mat_list = cp.asarray(temp_mat_list)

    binned_temp_mat_list = cp.zeros((len(temp_to_use), N, N_t), dtype=cp.int32)
    binned_spike_mat_list = cp.zeros((len(temp_to_use), N, M_t), dtype=cp.int32)
    binned_temp_mat_list_sq = cp.zeros((len(temp_to_use), N, N_t), dtype=cp.int32)
    binned_spike_mat_list_sq = cp.zeros((len(temp_to_use), N, M_t), dtype=cp.int32)

    num_corr = M_t - N_t + 1
    corr = cp.zeros((len(temp_to_use), num_corr), dtype=cp.float32)

    print("Init Done")

    # Calc binning here
    binned_temp_mat_list = cp.sum(temp_mat_list, axis = 3)
    binned_temp_mat_list_sq = binned_temp_mat_list ** 2

    # numba does not allow multi dim reduction
    c1 = (N_t * N) * cp.ones(len(temp_to_use))
    c2 = cp.sum(binned_temp_mat_list[:,:,:N_t], axis = (1,2), dtype=cp.int32)
    c3 = cp.sum(binned_temp_mat_list_sq[:,:,:N_t], axis = (1,2), dtype=cp.int32)

    c3 = c1 * c3 - c2 ** 2

    binned_spike_mat_list = cp.sum(spike_mat, axis = 3, dtype=cp.int32)
    binned_spike_mat_list_sq = binned_spike_mat_list ** 2

    #for t in range(N_t - 1, M_t):
    as_strided = cp.lib.stride_tricks.as_strided

    num_template = binned_temp_mat_list.shape[0]
    strides = binned_spike_mat_list.strides
    target_spike_mat_list = as_strided(binned_spike_mat_list, \
                                       shape = (num_corr, num_template, N, N_t), \
                                       strides = (strides[2], strides[0], strides[1], strides[2]))
    target_spike_mat_list_sq = as_strided(binned_spike_mat_list_sq, \
                                          shape = (num_corr, num_template, N, N_t), \
                                          strides = (strides[2], strides[0], strides[1], strides[2]))

    target_mul_list = binned_temp_mat_list * target_spike_mat_list
    S1 = cp.sum(target_mul_list, axis= (2,3), dtype=cp.int32)
    S2 = cp.sum(target_spike_mat_list, axis= (2,3), dtype=cp.int32)
    S3 = cp.sum(target_spike_mat_list_sq, axis= (2,3), dtype=cp.int32)

    x = ((c1 * S1 - c2 * S2) ** 2) / (c3 * (c1 * S3 - S2 ** 2))
    corr = x.transpose()

    corr = cp.asnumpy(corr)
    c1 = cp.asnumpy(c1)
    c2 = cp.asnumpy(c2)
    c3 = cp.asnumpy(c3)
    return corr, c1, c2, c3

def template_matching_gpu(spike_mat, temp_mat_list, bin_width, temp_to_use):
    print("Initiated template matching")

    N = temp_mat_list.shape[1]
    N_t = temp_mat_list.shape[2]
    M_t = spike_mat.shape[2]

    # cpu to gpu memcpy
    spike_mat = cp.asarray(spike_mat)
    temp_mat_list = cp.asarray(temp_mat_list)

    binned_temp_mat_list = np.zeros((len(temp_to_use), N, N_t), dtype=np.int32)
    binned_spike_mat_list = np.zeros((len(temp_to_use), N, M_t), dtype=np.int32)
    binned_temp_mat_list_sq = np.zeros((len(temp_to_use), N, N_t), dtype=np.int32)
    binned_spike_mat_list_sq = np.zeros((len(temp_to_use), N, M_t), dtype=np.int32)
    
    num_corr = M_t - N_t + 1
    corr = cp.zeros((len(temp_to_use), num_corr), dtype=cp.float32)

    this_corr = cp.zeros(num_corr)

    print("Init Done")

    # Calc binnig here
    binned_temp_mat_list = cp.sum(temp_mat_list, axis = 3)
    binned_temp_mat_list_sq = binned_temp_mat_list ** 2
    binned_spike_mat_list = cp.sum(spike_mat, axis = 3, dtype=cp.int32)
    binned_spike_mat_list_sq = binned_spike_mat_list ** 2

    as_strided = cp.lib.stride_tricks.as_strided

    for temp_idx in range(len(temp_to_use)):
        binned_temp_mat = binned_temp_mat_list[temp_idx]
        binned_spike_mat = binned_spike_mat_list[temp_idx]

        num_neuron = binned_spike_mat.shape[0]
        stride0 = binned_spike_mat.strides[0]
        stride1 = binned_spike_mat.strides[1]
        target_spike_mat = as_strided(binned_spike_mat, shape = (num_corr,num_neuron,N_t), strides = (stride1, stride0, stride1))
        target_spike_mat = target_spike_mat.reshape((num_corr, -1))
        binned_temp_mat = binned_temp_mat.flatten()
        x = cp.corrcoef(target_spike_mat, binned_temp_mat)
        this_corr = x[num_corr][:-1]

        this_corr = this_corr ** 2
        corr[temp_idx] = this_corr

    corr = cp.asnumpy(corr)
    return corr

# @profile
def main():
    # Load simulation parameters
    properties = configparser.ConfigParser()
    properties.read('tm.params')

    spike_data = properties.get('general','spike_data')
    template_data = properties.get('general','template_data')

    sim_duration = properties.getint('general','duration')
    sampling_rate = properties.getint('general','sampling_rate')

    template_window = properties.getint('template_generation','template_window') * sampling_rate

    temp_to_use = properties.get('template_matching','temp_to_use')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_cell_num', type=int, dest='target_cell_num', default=-1, help='target_cell_num')
    parser.add_argument('--bin_width', type=int, dest='bin_width', default=-1, help='bin_width')
    
    args = parser.parse_args()
    
    target_cell_num = args.target_cell_num
    bin_width = args.bin_width
    
    if target_cell_num < 0:
        target_cell_num = properties.getint('template_matching','target_cell_num')
    if bin_width < 0:
        bin_width = properties.getint('template_matching','bin_width')
    bin_width = int(bin_width * sampling_rate // 1000)

    sparse = properties.getboolean('general','gpu_sparse')
    chunk_size = properties.getint('general','chunk_size') * sampling_rate // 1000

    # Load spike & template data
    total_spike_mat = load_data(spike_data)
    temp_mat_list, temp_neuron_list = load_templates(template_data, template_window)

    cell_num, duration = total_spike_mat.shape

    duration = min(sim_duration, duration)
    total_spike_mat = total_spike_mat[:, :duration]
    duplicate_num = (target_cell_num - 1) // cell_num + 1

    nb_chunks = (duration-1)//chunk_size + 1

    if temp_to_use == 'all':
        temp_to_use = range(len(temp_mat_list))
    else:
        temp_to_use = eval(temp_to_use)

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
    print("gpu sparse :", sparse)
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
    print("firing rate:", np.count_nonzero(total_spike_mat) / original_cell_num / duration * sampling_rate, \
            "->", np.count_nonzero(new_total_spike_mat) / cell_num / duration * sampling_rate)

    # for temp_neu in temp_neuron_list:
    #     print(temp_neu)
    
    # Check if all the elements contain the same list
    prev_neu = np.array([-1])
    for temp_id in temp_to_use:
        if (prev_neu != -1).all():
            assert(list(prev_neu) == list(temp_neuron_list[temp_id]))
        prev_neu = temp_neuron_list[temp_id]
    
    new_total_spike_mat = np.array(new_total_spike_mat, dtype = np.int32)
    np.save('new_total_spike_mat.npy', new_total_spike_mat)

    new_temp_mat_list = np.array(new_temp_mat_list, dtype = np.int32)
    temp_to_use = np.array(temp_to_use, dtype = np.int32)
    temp_neuron_list = np.array(temp_neuron_list, dtype = np.int32)

    # bin and duplicate the data for preprocessing
    N = new_temp_mat_list[0].shape[0]
    N_t = int(new_temp_mat_list[0].shape[-1] / bin_width)
    M_t = int(new_total_spike_mat.shape[-1] / bin_width)
    N_tm = len(temp_to_use)
    
    del new_total_spike_mat

    temp_mat_list = np.zeros((len(temp_to_use), N, N_t, bin_width), dtype=np.int32)

    for temp_idx in range(temp_to_use.shape[0]):
        temp_mat = new_temp_mat_list[temp_to_use[temp_idx]]
        temp_neu = temp_neuron_list[temp_to_use[temp_idx]] 

        temp_mat_list[temp_idx] = temp_mat[temp_neu, :N_t * bin_width].reshape(len(temp_neu), -1, bin_width)
    
    # spike_mat_list = new_total_spike_mat[:, :M_t * bin_width].reshape(N, -1, bin_width)

    print("\nSTART BENCHMARK")
    time_sum = 0
    num_iter = 100
    for i in range(num_iter):
        corr, c1, c2, c3, elapsed = template_matching_gpu_realtime(temp_mat_list, bin_width, temp_to_use, sparse, nb_chunks, chunk_size, M_t)
        time_sum += elapsed
        print(f"Elapsed: {elapsed} (s)")
    print("\nEND BENCHMARK")
    print("Iter:", num_iter)
    print(f"Average per iter: {time_sum / num_iter} (s)")

    #corr = template_matching_gpu_static(spike_mat_list, temp_mat_list, bin_width, temp_to_use)
    #corr = template_matching_gpu(spike_mat_list, temp_mat_list, bin_width, temp_to_use)

    debug_file = open('multicore_spike_out.dat', 'w')

    for temp_id in range(corr.shape[0]):
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
    
if __name__ == '__main__':
    main()
