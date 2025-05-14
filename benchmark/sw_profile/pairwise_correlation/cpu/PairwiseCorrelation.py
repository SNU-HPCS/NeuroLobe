import sys, os
import numpy as np

import time
import timeit
import scipy

import cProfile
import yappi

from scipy.ndimage import gaussian_filter1d
from neo.core import AnalogSignal
import quantities as pq
from elephant import spike_train_generation
import configparser
import correlate
import networkx as nx 
import argparse

np.random.seed(0)
debug = False

def load_data(dataset):
    data = np.load(dataset, allow_pickle=True)
    total_spike_mat = np.transpose(data['input'])

    return total_spike_mat
                            
def pairwise_correlation_cpu(M_t, spike_per_ts, cell_num, duration, window,
                             corr_period, sparse_conn_mat, conn_forward_ind,
                             conn_total, correlogram, spike_hist, spike_squared,
                             num_thread, i, inner_profile):
    
    # Calculate pairwise correlation with partial correlograms
    for t in range(M_t):
        # get the list of neurons spiked at the given ts
        spike_list = np.array(spike_per_ts[t], dtype=np.uint32)
        num_spikes = spike_list.shape[0]

        correlate.pairwise_correlation_cpu_inner(spike_hist, spike_squared, spike_list, conn_forward_ind,
                                                 sparse_conn_mat,
                                                 correlogram, cell_num, num_spikes, conn_total,
                                                 corr_period, t, window + 1, num_thread)
        

#@profile
if __name__ == '__main__':
    # Load simulation parameters
    properties = configparser.ConfigParser()
    properties.read('pc.params')

    dataset = properties.get('general','dataset')

    sim_duration = properties.getint('general','duration')
    sampling_rate = properties.getint('general','sampling_rate')

    window = properties.getint('pairwise_correlation','window')

    # NOTE : parse argument from the command
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_cell_num', type=int, dest='target_cell_num', default=-1, help='target_cell_num')
    parser.add_argument('--epsilon', type=float, dest='epsilon', default=-1, help='epsilon')
    parser.add_argument('--num_thread', type=int, dest='num_thread', default=-1, help='num_thread')
    parser.add_argument('--bin_width', type=int, dest='bin_width', default=-1, help='bin_width')
    parser.add_argument('--corr_period', type=int, dest='corr_period', default=-1, help='corr_period')
    args = parser.parse_args()

    target_cell_num = args.target_cell_num
    epsilon         = args.epsilon
    num_thread      = args.num_thread
    bin_width       = args.bin_width
    corr_period     = args.corr_period

    #sim_num_workers = properties.getint('general','num_workers')
    use_gpu = properties.getboolean('general','use_gpu')
    PARSE_BENCHMARK = properties.getboolean('general','parse_benchmark')

    beta = properties.getfloat('network_config', 'beta')

    if target_cell_num < 0:
        target_cell_num = properties.getint('pairwise_correlation','target_cell_num')
    if epsilon < 0:
        epsilon = properties.getfloat('network_config', 'epsilon')
    if num_thread < 0:
        num_thread = properties.getfloat('general', 'num_workers')
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
    print("bin width:", bin_width, "(timesteps)")
    print("window:", window, "(bins)")
    print("sampling rate:", sampling_rate)
    print(f"cpus : {num_thread}")
    print(f"use gpu : {use_gpu}")
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
    
    np.save('new_total_spike_mat.npy',new_total_spike_mat)
    del new_total_spike_mat

    # Make random connection between neurons
    conn_list = []
    np.random.seed(0)
    conn = nx.watts_strogatz_graph(cell_num, int(cell_num * epsilon), beta, seed=0).edges()
    for src, dst in conn:
        if src != dst:
            conn_list.append([src, dst])
            conn_list.append([dst, src])
    conn_list = sorted(conn_list)

    sparse_conn_mat = np.array(conn_list,dtype=np.uint32)
    

    new_total_spike_mat = np.load('new_total_spike_mat.npy',allow_pickle=True,mmap_mode='r')
    spike_mat = new_total_spike_mat.reshape(cell_num, -1,bin_width)

    # convert this to spike idx per ts
    print('Spike Compression Start')
    spike_per_ts = [[] for _ in range(M_t)]
    for t in range(M_t):
        for b in range(spike_mat.shape[2]):
            for neu in range(cell_num):
                if spike_mat[neu,t,b] > 0:
                    spike_per_ts[t].append(neu)
    print('Spike Compression Done')

    conn_forward_ind = np.zeros((cell_num, 2), dtype=np.uint32)
    src_prev = sparse_conn_mat[0][0]
    conn_forward_ind[src_prev][0] = 0
    for idx in range(len(sparse_conn_mat)):
        src, dst = sparse_conn_mat[idx]
        if src != src_prev:
            conn_forward_ind[src_prev][1] = idx
            conn_forward_ind[src][0] = idx
        src_prev = src
    conn_forward_ind[src][1] = len(sparse_conn_mat)

    conn_total = sparse_conn_mat.shape[0]
    correlogram = np.zeros((conn_total, window + 1), dtype=np.uint32)

    # history of the spiking activity
    spike_hist = np.zeros((cell_num, window + 1), dtype=np.uint32)
    spike_squared = np.zeros(cell_num, dtype=np.uint32)

    time_sum = 0
    num_iter = 1

    inner_profile=0
    yappi.set_clock_type("wall")
    yappi.start()
    for i in range(num_iter):
        tic=timeit.default_timer()
        # Iterate multiple times
        pairwise_correlation_cpu(M_t, spike_per_ts, cell_num, duration / bin_width,
                                 window, corr_period, sparse_conn_mat, conn_forward_ind,
                                 conn_total, correlogram, spike_hist, spike_squared,
                                 num_thread, i, inner_profile)
        toc=timeit.default_timer()
        time_sum += toc - tic
        print(f"Elapsed: {toc - tic} (s)")
    yappi.stop()

    threads = yappi.get_thread_stats()
    for thread in threads:
        print("Function stats for (%s) (%d)"%(thread.name, thread.id))
        yappi.get_func_stats(ctx_id=thread.id).print_all()

    print(f"Num iter: {num_iter}")
    print(f"Average per iter: {time_sum / num_iter} (s)")
    average = time_sum / M_t * 1000 / num_iter
    print(f"Average per ts: {average} (ms)")

    if debug:
        print()
        for i in range(cell_num):
            print(i, end=": ")
            print(pcc[i])
