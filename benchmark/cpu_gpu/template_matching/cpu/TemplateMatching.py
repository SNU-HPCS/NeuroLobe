import sys
import numpy as np

import time
import timeit

import scipy

from scipy.ndimage import gaussian_filter1d
from neo.core import AnalogSignal
import quantities as pq
from elephant import spike_train_generation
import configparser
import correlate
import cProfile
import yappi
import argparse

np.random.seed(0)

def load_data(dataset):
    data = np.load(dataset, allow_pickle=True)
    total_spike_mat = np.transpose(data['input'])

    return total_spike_mat

def load_templates(dataset, template_window):
    data = np.load(dataset, allow_pickle=True, mmap_mode='r')
    
    temp_data = data['template'][()]

    temp_mat_list = []
    temp_neuron_list = []
    for temp_id in temp_data:
        template = np.transpose(temp_data[temp_id]).astype(int)
        assert(template.shape[1] > template_window)
        temp_mat_list.append(template[:, :template_window])
        temp_neuron_list.append(np.arange(template.shape[0], dtype=np.uint32))
    return temp_mat_list, temp_neuron_list


def template_matching_cpu(spike_per_ts, temp_mat_list, bin_width, temp_to_use, sim_num_workers, sum_time, i, inner_profile):
    # print("Initiated template matching")
    cell_num = temp_mat_list.shape[1]
    template_width = temp_mat_list.shape[2]
    M_t = len(spike_per_ts)
    num_templates = len(temp_to_use)

    temp_mat = np.zeros((num_templates, cell_num, template_width), dtype=np.uint32)
    temp_mat_sq = np.zeros((num_templates, cell_num, template_width), dtype=np.uint32)

    spike_hist = np.zeros((cell_num, template_width + 1), dtype=np.uint32)
    S1_partial_hist = np.zeros((num_templates, template_width), dtype=np.uint)

    num_corr = M_t - template_width + 1
    corr = np.zeros((num_templates, num_corr), dtype=np.float32)
    
    temp_mat[:,:,:] = np.sum(temp_mat_list[:,:,:],axis=3)
    temp_mat_sq[:,:,:] = temp_mat[:,:,:] ** 2

    c1 = (template_width * cell_num) * np.ones(num_templates)
    c2 = np.sum(temp_mat[:,:,:template_width], axis = 2, dtype=np.uint)
    c2 = np.sum(c2, axis = 1, dtype=np.uint)
    c3 = np.sum(temp_mat_sq[:,:,:template_width], axis = 2, dtype=np.uint)
    c3 = np.sum(c3, axis = 1, dtype=np.uint)
    c3 = c1 * c3 - c2 ** 2
    c1 = np.array(c1, dtype=np.float32)
    c2 = np.array(c2, dtype=np.float32)
    c3 = np.array(c3, dtype=np.float32)

    S1 = np.zeros(template_width, dtype=np.uint)
    S2 = np.zeros(1, dtype=np.uint32)
    S3 = np.zeros(1, dtype=np.uint32)


    tic=timeit.default_timer()
    for t in range(M_t):
        spike_list = np.array(spike_per_ts[t], dtype=np.uint32)
        correlate.template_matching_cpu_sparse_inner(spike_list,
                                                    spike_hist,
                                                    S1_partial_hist,
                                                    temp_mat,
                                                    c1, c2, c3,
                                                    S1, S2, S3,
                                                    corr, t, num_templates,
                                                    cell_num, bin_width, template_width,
                                                    sim_num_workers)
        #    inner_profile.write(str(inner_toc-inner_tic)+'\n')
    

    toc=timeit.default_timer()
    sum_time += (toc-tic)
    average = (toc - tic) / M_t * 1000
    print(f"Elapsed: {toc - tic} (s)")

    return corr, c1, c2, c3, sum_time


def template_matching_cpu_inner(cur_spike,
                                spike_hist,
                                temp_mat,
                                c1, c2, c3,
                                S2, S3,
                                corr, t, num_templates,
                                cell_num, bin_width, template_width):

    # dynamically bin the spikes here
    for n in range(cell_num):
        spike_hist[n, t % (template_width + 1)] = 0
    for b in range(bin_width):
        for n in range(cell_num):
            spike_hist[n, t % (template_width + 1)] += cur_spike[b, n]

    for temp_idx in range(num_templates):
        S1 = 0
        for n in range(cell_num):
            for t_shift in range(template_width):
                template = temp_mat[temp_idx,n,t_shift]
                spike = spike_hist[n,(t+2+t_shift)%(template_width + 1)]
                S1 = S1 + template * spike

            cur_spike = spike_hist[n, t % (template_width + 1)]
            prev_spike = spike_hist[n, (t-template_width) % (template_width + 1)]
            S2[temp_idx] = S2[temp_idx] + cur_spike - prev_spike
            S3[temp_idx] = S3[temp_idx] + cur_spike**2 - prev_spike ** 2

        x = ((c1[temp_idx] * S1 - c2[temp_idx] * S2[temp_idx]) ** 2) / (c3[temp_idx] * (c1[temp_idx] * S3[temp_idx] - S2[temp_idx] ** 2))
        if t >= template_width - 1:
            corr[temp_idx,t - template_width + 1] = x

if __name__ == '__main__':
    # Load simulation parameters
    properties = configparser.ConfigParser()
    properties.read('tm.params')

    dataset = properties.get('general','dataset')
    template_dataset = properties.get('general','template_dataset')

    sim_duration = properties.getint('general','duration')
    sampling_rate = properties.getint('general','sampling_rate')

    template_window = properties.getint('template_generation','template_window') * sampling_rate

    # NOTE : parse argument from the command
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_cell_num', type=int, dest='target_cell_num', default=-1, help='target_cell_num')
    parser.add_argument('--num_thread', type=int, dest='num_thread', default=-1, help='num_thread')
    parser.add_argument('--bin_width', type=int, dest='bin_width', default=-1, help='bin_width')
    
    args = parser.parse_args()
    
    target_cell_num = args.target_cell_num
    sim_num_workers = args.num_thread
    bin_width = args.bin_width
    
    if target_cell_num < 0:
        target_cell_num = properties.getint('template_matching','target_cell_num')
    if sim_num_workers <0:
        sim_num_workers = properties.getint('general','num_workers')
    if bin_width < 0:
        bin_width = properties.getint('template_matching','bin_width')
    bin_width = int(bin_width * sampling_rate // 1000)

    # duplicate_num = properties.getint('template_matching','duplicate_num')
    temp_to_use = properties.get('template_matching','temp_to_use')
    
    USE_GPU = properties.getboolean('general','use_gpu')
    PARSE_BENCHMARK = properties.getboolean('general','parse_benchmark')

    # Load spike data
    total_spike_mat = load_data(dataset)
    temp_mat_list, temp_neuron_list = load_templates(template_dataset, template_window)

    cell_num, duration = total_spike_mat.shape
    duration = min(sim_duration, duration)
    total_spike_mat = total_spike_mat[:, :duration]
    duplicate_num = (target_cell_num - 1) // cell_num + 1
    
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
    print("bin width:", bin_width, "(samples)")
    print("template window:", template_window, "(samples)")
    print("sampling rate:", sampling_rate)
    print("temp to use:", temp_to_use)
    print("cpus :", sim_num_workers)
    print("use gpu :", USE_GPU)
    print("parse benchmark :", PARSE_BENCHMARK)
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
    del generated_temps
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
    
    # Check if all the elements contain the same list
    prev_neu = np.array([-1])
    for temp_id in temp_to_use:
        if (prev_neu != -1).all():
            assert(list(prev_neu) == list(temp_neuron_list[temp_id]))
        prev_neu = temp_neuron_list[temp_id]

    new_total_spike_mat = np.array(new_total_spike_mat, dtype = np.uint32)
    new_temp_mat_list = np.array(new_temp_mat_list, dtype = np.uint32)
    temp_to_use = np.array(temp_to_use, dtype = np.int32)
    temp_neuron_list = np.array(temp_neuron_list, dtype = np.uint32)

    # bin and duplicate the data for preprocessing
    N = new_temp_mat_list[0].shape[0]
    N_t = int(new_temp_mat_list[0].shape[-1] / bin_width)
    M_t = int(new_total_spike_mat.shape[1] / bin_width)
    N_tm = len(temp_to_use)
   
    np.save('total_spike_mat.npy', new_total_spike_mat)
    del new_total_spike_mat
    
    temp_mat_list = np.zeros((len(temp_to_use), N, N_t, bin_width), dtype=np.uint32)
    for temp_idx in range(temp_to_use.shape[0]):
        temp_mat = new_temp_mat_list[temp_to_use[temp_idx]]
        temp_neu = temp_neuron_list[temp_to_use[temp_idx]]

        temp_mat_list[temp_idx] = temp_mat[temp_neu, :N_t * bin_width].reshape(len(temp_neu), -1, bin_width)

    # This is for compiling only
    compile_time = 0
    new_total_spike_mat = np.load('total_spike_mat.npy',allow_pickle=True,mmap_mode='r')
    spike_mat = new_total_spike_mat.reshape(N,-1,bin_width)
    M_t = spike_mat.shape[1]
    print('Spike Compression Start')
    spike_per_ts = [[] for _ in range(M_t)]
    for t in range(M_t):
        for b in range(spike_mat.shape[2]):
            for neu in range(cell_num):
                if spike_mat[neu,t,b] > 0:
                    spike_per_ts[t].append(neu)
    print('Spike Compression Done')
    
    # with profiler
    sum_time = 0
    num_iter = 10
    # with cProfile
    #with cProfile.Profile() as pr:
    #    for i in range(num_iter):
    #        corr, c1, c2, c3, sum_time = template_matching_cpu(spike_per_ts, temp_mat_list, bin_width, temp_to_use, sim_num_workers, sum_time)
    #
    #    pr.print_stats()

    #inner_profile = open('tm_inner_profile.dat','w')
    inner_profile = 0
    # with yappi
    yappi.set_clock_type("wall")
    yappi.start(profile_threads=True)
    for i in range(num_iter):
        corr, c1, c2, c3, sum_time = template_matching_cpu(spike_per_ts, temp_mat_list, bin_width, temp_to_use, sim_num_workers, sum_time, i, inner_profile)
    yappi.stop()

    threads = yappi.get_thread_stats()
    for thread in threads:
        print("Function stats for (%s) (%d)"%(thread.name, thread.id))
        yappi.get_func_stats(ctx_id=thread.id).print_all()


    # without profiler
    # corr, c1, c2, c3 = template_matching_cpu(spike_per_ts, temp_mat_list, bin_width, temp_to_use, sim_num_workers)
    print(f"Num iter: {num_iter}")
    print(f"Average per iter: {sum_time / num_iter} (s)")
    average = sum_time / M_t * 1000 / num_iter
    print(f"Average per ts: {average} (ms)")

    #toc=timeit.default_timer()
    #print(f"Elapsed: {toc - tic} (s)")
    
    # NOTE : debug file commented out
    debug_file = open('debug.dat', 'w')
    for temp_id in range(corr.shape[0]):
        debug_corr = np.array(corr[temp_id])
        show_corr = debug_corr[np.logical_not(np.isnan(debug_corr))]
        show_index = np.argwhere(np.logical_not(np.isnan(debug_corr)))
        result_list = [[corr, index[-1]] for corr, index in zip(show_corr, show_index)]
        debug_file.write(str(temp_id) + ": " + str(result_list) + "\n")

    debug_file.close()
