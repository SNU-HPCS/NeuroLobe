import sys
import numpy as np
import cupy as cp

import time
import timeit

from scipy.ndimage import gaussian_filter1d
from neo.core import AnalogSignal
import quantities as pq
from elephant import spike_train_generation
import configparser
from cupyx.profiler import benchmark
import networkx as nx 
from scipy.ndimage import gaussian_filter1d
import argparse


np.random.seed(0)
debug = False

def load_data(dataset):
    data = np.load(dataset, allow_pickle=True)
    total_spike_mat = np.transpose(data['input'])

    return total_spike_mat

def pairwise_correlation_gpu_realtime(spike_mat, cell_num, duration, window, corr_period, sparse_conn_mat, chunk_size, sparse):
    M_t = spike_mat.shape[1]
    N_t = window + 1

    # cpu to gpu memcpy
    spike_mat = cp.asarray(spike_mat, dtype=cp.uint32)
    sparse_conn_mat = cp.asarray(sparse_conn_mat)
    conn_total = sparse_conn_mat.shape[0]
    # pcc = cp.zeros((conn_total, M_t // corr_period, window + 1), dtype=cp.float32)
    binned_spike_mat = cp.sum(spike_mat, axis = 2, dtype=cp.float32)
    partial_corr = cp.zeros((conn_total, window + 1), dtype=cp.float32)
    num_spikes = cp.zeros(cell_num, dtype=cp.uint32) # Number of spikes so far
    spike_total = binned_spike_mat.shape[1]
    assert(chunk_size > window)
    nb_chunks = (binned_spike_mat.shape[1]-1) // (chunk_size - window) + 1
    if chunk_size == M_t-window:
        nb_chunks = 1


    if debug:
        debug_pcc = [[] for _ in range(cell_num)]

    tic=timeit.default_timer()

    for chunk_idx in range(nb_chunks):
        offset = chunk_idx*(chunk_size - window) + window
        spike_mat = binned_spike_mat[:,(offset - window):(offset + chunk_size - window)]

        # print(spike_total - window)
        for t in range(spike_total - window):
            # print((chunk_idx*(chunk_size - window) + t), end=" ")
            # Spike history at target neurons
            spike_target = binned_spike_mat[:, t:(t+window+1)]
            # if sparse:
            #     spike_target = cp.sparse.csr_matrix(binned_spike_mat[:, t:(t+window+1)])
            # else:
            #     spike_target = binned_spike_mat[:, t:(t+window+1)]

            ref = spike_target[:, -1]  # Extract spikes for all neurons at time t

            # if not ref.any():
            #     continue
    
            num_spikes += cp.square(ref, dtype=cp.uint32)
            # print(num_spikes)
            # if sparse:
            #     num_spikes += ref.multiply(ref)
            # else:
            #     num_spikes += cp.square(ref, dtype=cp.uint32)

            # stretch ref matrix
            spike_reference = cp.repeat(ref[:, cp.newaxis] , window+1, axis=1)
            # if sparse:
            #     spike_reference = window_ones.multiply(ref)
            # else:
            #     spike_reference = cp.repeat(ref[:, cp.newaxis] , window+1, axis=1)

            # Calculate the cross-correlation and update partial_corr
            if sparse:
                sparse_spike_reference = cp.sparse.csr_matrix(spike_reference[sparse_conn_mat[:,0], :])
                sparse_spike_target = cp.sparse.csr_matrix(spike_target[sparse_conn_mat[:,1], :])
                # print(cp.count_nonzero(spike_reference[sparse_conn_mat[:,0], :]), spike_reference[sparse_conn_mat[:,0], :].size)
                # print(cp.count_nonzero(spike_target[sparse_conn_mat[:,1], :]), spike_target[sparse_conn_mat[:,1], :].size)

            if sparse:
                partial_corr += sparse_spike_reference.multiply(sparse_spike_target).todense()
            else:
                partial_corr += cp.multiply(spike_reference[sparse_conn_mat[:,0], :], spike_target[sparse_conn_mat[:,1], :])

            if (chunk_idx*(chunk_size - window) + t) % corr_period == 0:
                # Calculate correlations
                norm = num_spikes[sparse_conn_mat[:,0]] * num_spikes[sparse_conn_mat[:,1]]
                norm = cp.sqrt(norm)
                pcc = cp.flip(partial_corr, axis=1) / norm[:, cp.newaxis]
                # if sparse:
                #     norm = num_spikes[sparse_conn_mat[:,0]].multiply(num_spikes[sparse_conn_mat[:,1]])
                #     norm = norm.sqrt()
                #     pcc = cp.flip(partial_corr, axis=1).multiply(conn_ones / norm)

                # else:
                #     norm = num_spikes[sparse_conn_mat[:,0]] * num_spikes[sparse_conn_mat[:,1]]
                #     norm = cp.sqrt(norm)
                #     pcc = cp.flip(partial_corr, axis=1) / norm[:, cp.newaxis]

                if debug:
                    for target in range(cell_num):
                        debug_pcc[target].append({})
                    for conn_idx in range(conn_total):
                        reference, target = sparse_conn_mat[conn_idx]
                        if cp.isfinite(pcc[conn_idx]).all() and cp.sum(pcc[conn_idx]) > 0:
                            debug_pcc[int(target)][-1][int(reference)] = pcc[conn_idx].tolist()
                        # if sparse:
                        #     if sum(cp.nan_to_num(pcc[conn_idx].todense())) > 0:
                        #         debug_pcc[int(target)][-1][int(reference)] = pcc[conn_idx].todense().tolist()
                        # else:
                        #     if sum(cp.nan_to_num(pcc[conn_idx])) > 0:
                        #         debug_pcc[int(target)][-1][int(reference)] = pcc[conn_idx].tolist()

    toc=timeit.default_timer()

    if debug:
        debug_file = open('multicore_spike_out.dat', 'w')
        for i in range(cell_num):
            debug_file.write(str(i) + ": " + str(debug_pcc[i]) + "\n")
        debug_file.close()

    return toc - tic   


if __name__ == '__main__':
    # Load simulation parameters
    properties = configparser.ConfigParser()
    properties.read('pc.params')

    dataset = properties.get('general','dataset')

    sim_duration = properties.getint('general','duration')
    sampling_rate = properties.getint('general','sampling_rate')

    window = properties.getint('pairwise_correlation','window')

    sim_num_gpu = properties.getint('general','num_gpu')
    sparse = properties.getboolean('general','gpu_sparse')

    beta = properties.getfloat('network_config','beta')

    chunk_size = properties.getint('general','chunk_size') * sampling_rate // 1000

    parser = argparse.ArgumentParser()
    parser.add_argument('--target_cell_num', type=int, dest='target_cell_num', default=-1, help='target_cell_num')
    parser.add_argument('--epsilon', type=float, dest='epsilon', default=-1, help='epsilon')
    parser.add_argument('--bin_width', type=int, dest='bin_width', default=-1, help='bin_width')
    parser.add_argument('--corr_period', type=int, dest='corr_period', default=-1, help='corr_period')
    args = parser.parse_args()

    target_cell_num = args.target_cell_num
    epsilon         = args.epsilon
    bin_width       = args.bin_width
    corr_period     = args.corr_period

    if target_cell_num < 0:
        target_cell_num = properties.getint('pairwise_correlation','target_cell_num')
    if epsilon < 0:
        epsilon = properties.getfloat('network_config', 'epsilon')
    if bin_width < 0:
        bin_width  = properties.getfloat('pairwise_correlation', 'bin_width')
    if corr_period < 0:
        corr_period = properties.getfloat('pairwise_correlation', 'corr_period')
    bin_width = int(bin_width * sampling_rate // 1000)
    # Load spike data
    total_spike_mat = load_data(dataset)

    cell_num, duration = total_spike_mat.shape
    duration = min(sim_duration, duration)
    total_spike_mat = total_spike_mat[:, :duration]
    duplicate_num = target_cell_num // cell_num + 1

    print("=====================================")
    print("duration:", duration)
    print("cell num:", cell_num)
    print("duplicate num:", duplicate_num)
    print("target cell num:", target_cell_num)
    print("bin width:", bin_width, "(ts)")
    print("correlation period", corr_period)
    print("window:", window, "(bins)")
    print("sampling rate:", sampling_rate)
    print("epsilon:", epsilon)
    print("beta:", beta)
    print(f"gpu sparse : {sparse}")
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
        del generated_spikes

    original_cell_num = cell_num
    cell_num = cell_num * duplicate_num
    
    print("\nduplicate done by", duplicate_num)
    
    if target_cell_num != -1:
        assert(target_cell_num <= cell_num)
        cell_num = target_cell_num
        new_total_spike_mat = new_total_spike_mat[:target_cell_num, :]
        
        print("removed cells to match target cell num")

    print("new cell num:", cell_num)
    print("firing rate:", np.count_nonzero(total_spike_mat) / original_cell_num / duration * sampling_rate, \
            "->", np.count_nonzero(new_total_spike_mat) / cell_num / duration * sampling_rate)


    # perform pairwise correlation in SW
    new_total_spike_mat = np.array(new_total_spike_mat, dtype = np.int32)
    M_t = int(new_total_spike_mat.shape[-1] / bin_width)

    spike_mat = new_total_spike_mat[:, :M_t * bin_width].reshape(cell_num, -1, bin_width)
    padded_spike_mat = np.concatenate((np.zeros([cell_num, window, bin_width]), spike_mat), axis=1)
    assert(chunk_size % bin_width == 0)
    chunk_size = chunk_size // bin_width

    del new_total_spike_mat
    del spike_mat

    # Make random connection between neurons
    conn_list = []
    np.random.seed(0)
    conn = nx.watts_strogatz_graph(cell_num, int(cell_num * epsilon), beta, seed=0).edges()
    for src, dst in conn:
        if src != dst:
            conn_list.append([src, dst])
            conn_list.append([dst, src])
    conn_list = sorted(conn_list)
    del conn
    sparse_conn_mat = np.array(conn_list,dtype=np.uint32)
    print(sparse_conn_mat.shape)

    time_sum = 0
    num_iter = 10
    print("START BENCHMARK")
    for i in range(num_iter):
        benchmark_time = pairwise_correlation_gpu_realtime(padded_spike_mat, cell_num, duration / bin_width, window, corr_period, sparse_conn_mat, chunk_size, sparse)
        print(f"Elapsed: {benchmark_time} (s)")
        time_sum += benchmark_time
    print("END BENCHMARK")
    print("Iter: ", num_iter)
    print(f"Average per iter: {time_sum/num_iter} (s)")
    del padded_spike_mat
    # for chunk_idx in range(nb_chunks):
    #     offset = chunk_idx*(chunk_size - window * bin_width) + window * bin_width
    #     print(offset - window * bin_width, offset + chunk_size - window * bin_width)
    #     padded_spike_mat = np.load('padded_spike_mat.npy',allow_pickle=True,mmap_mode='r')
    #     spike_mat = padded_spike_mat[:,(offset - window * bin_width):(offset + chunk_size - window * bin_width)].reshape(cell_num, -1,bin_width)
    #     benchmark_time = pairwise_correlation_gpu_realtime(spike_mat, cell_num, duration / bin_width, window, corr_period, sparse_conn_mat, chunk_idx, chunk_size // bin_width, sparse)
    #     del spike_mat
    #     del padded_spike_mat
    
