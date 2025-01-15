import sys, os
import numpy as np
import time
import timeit

import cProfile
import yappi

import configparser
import correlate
import argparse

np.random.seed(0)
debug = True

# NOTE : epuivalent to process_one_epoch
def FC_cpu(M_t, spike_per_ts, weight_per_spike,
                             conn_mem_1_weight_transpose,
                             conn_mem_2_weight_transpose,
                             I_t_vector1, I_t_vector2, i_dim, h_dim, o_dim, num_thread, bin_width):

    time_sum = 0
    num_iter = 10

    output_v_t = [None for _ in range(M_t)]

    yappi.set_clock_type("wall")
    yappi.start()
    for iteration in range(num_iter):
        elapsed_time = 0
        hidden_vector = np.zeros(h_dim, dtype=np.float32)
        weight_list_1 = np.zeros(h_dim, dtype=np.float32)
        weight_list_2 = np.zeros(o_dim, dtype=np.float32)

        spike_buf = np.zeros((i_dim, bin_width), dtype=np.float32)
        spike_sum = np.zeros(i_dim, dtype=np.float32)
        for t in range(M_t): # iterate over 600ms
            spike_list = np.array(spike_per_ts[t], dtype=np.int32)
            spike_weight = np.array(weight_per_spike[t], dtype=np.float32)
            num_spikes = spike_list.shape[0]

            output_vector = np.zeros(o_dim, dtype=np.float32)

            tic=timeit.default_timer()

            correlate.FC_cpu_inner(spike_list, spike_weight, conn_mem_1_weight_transpose, conn_mem_2_weight_transpose, hidden_vector, output_vector, weight_list_1, weight_list_2, I_t_vector1, I_t_vector2, spike_buf, spike_sum, i_dim, h_dim, o_dim, t, num_spikes, num_thread, bin_width)

            toc=timeit.default_timer()
            elapsed_time += toc - tic

            output_v_t[t] = output_vector.copy()

        time_sum += elapsed_time
        print("Iter: ", iteration)
        print(f"Elapsed: {elapsed_time} (s)")
    yappi.stop()

    threads = yappi.get_thread_stats()
    for thread in threads:
        print("Function stats for (%s) (%d)"%(thread.name, thread.id))
        yappi.get_func_stats(ctx_id=thread.id).print_all()


    print(f"Num iter: {num_iter}")
    print(f"Average per iter: {time_sum / num_iter} (s)")

    result_file = open('output.txt', 'w')
    for v_t_line in output_v_t:
        result_file.write(str(list(v_t_line)) + '\n')
        # result_file.write(str(list(v_t_line)) + '\n')
    result_file.close()

    average = time_sum / M_t * 1000 / num_iter
    print(f"Average per ts: {average} (ms)")

#@profile
if __name__ == '__main__':
    # Load simulation parameters
    properties = configparser.ConfigParser()
    properties.read('fc.params')

    dataset_dir = properties.get('general','dataset_dir')
    bin_width   = properties.getint('general','bin_width')

    sim_duration = properties.getint('general','duration')

    # NOTE : parse argument from the command
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_cell_num', type=int, dest='target_cell_num', default=-1, help='target_cell_num')
    parser.add_argument('--num_thread', type=int, dest='num_thread', default=-1, help='num_thread')
    args = parser.parse_args()

    target_cell_num = args.target_cell_num
    num_thread      = args.num_thread

    if target_cell_num < 0:
        target_cell_num = properties.getint('general','target_cell_num')
    if num_thread < 0:
        num_thread = properties.getfloat('general', 'num_thread')

    hidden_dimension = properties.getint('general', 'h_dim')

    # 1. load external spike data
    external_stimulus = np.load(dataset_dir+'external_stimulus.npy', allow_pickle=True)
    network_params = np.load(dataset_dir+'network_parameter.npy', allow_pickle=True)

    print(network_params)
    duration = network_params.item().get('simtime')

    if sim_duration > 0:
        duration = min(duration, sim_duration)

    h_dim = hidden_dimension
    o_dim = 2
    M_t = duration

    external_dim = target_cell_num
    external_offset = h_dim + o_dim
    num_neuron = h_dim + o_dim

    print("=====================================")
    print("M_t:", M_t)
    print("target cell num:", target_cell_num)
    print("h_dim:", h_dim)
    print("o_dim:", o_dim)
    print(f"cpus : {num_thread}")
    print("=====================================")

    # convert this to spike idx per ts
    print('Spike Compression Start')
    spike_per_ts = [[] for _ in range(M_t)]
    weight_per_spike = [[] for _ in range(M_t)]

    for spike in external_stimulus:
        timestep = int(spike[0])
        neuron_id = int(spike[1])
        spike_value = spike[2]
        if timestep >= M_t:
            break
        if spike_value != 0:
            spike_per_ts[timestep].append(neuron_id)
            weight_per_spike[timestep].append(spike_value)
    print('Spike Compression Done')

    # 2. load connection data
    conn_dict = np.load(dataset_dir+'connection.npy', allow_pickle=True)

    conn_mem_1_temp = conn_dict[1][0] # bci_neuron to neuron
    conn_mem_1_len = len(conn_mem_1_temp)
    conn_mem_2_temp = conn_dict[0][0] # neuron to neuron
    conn_mem_2_len = len(conn_mem_2_temp)

    # build conn_forward
    conn_forward_1 = np.zeros((external_dim, 2), dtype = np.int64)
    conn_forward_2 = np.zeros((h_dim + o_dim, 2), dtype = np.int64)

    conn_mem_1_weight_transpose = np.zeros(external_dim * h_dim, dtype = np.float32)
    conn_mem_2_weight_transpose = np.zeros(h_dim * o_dim, dtype = np.float32)

    print("Build connection")

    # 1. bci neuron to neuron connection
    for conn_idx in range(len(conn_mem_1_temp)):
        conn = conn_mem_1_temp[conn_idx]
        src = int(conn[0])
        dst = int(conn[1])
        has_entry = bool(conn[2])
        entry = conn[3]

        weight = entry[0][1]
        delay = entry[1][1]

        conn_mem_1_weight_transpose[dst * external_dim + src] = weight

    # 2. neuron to neuron connection
    for conn_idx in range(len(conn_mem_2_temp)):
        conn = conn_mem_2_temp[conn_idx]
        src = int(conn[0])
        dst = int(conn[1])
        has_entry = bool(conn[2])
        entry = conn[3]

        weight = entry[0][1]
        delay = entry[1][1]

        dst = dst - h_dim # subtract offset
        conn_mem_2_weight_transpose[dst * h_dim + src] = weight

    print("Build connection done")

    # 3. required states & parameters for FC
    print("Load states and parameters")
    state_dict = np.load(dataset_dir+'initial_states.npy', allow_pickle=True).item()

    I_t_list = np.array([state_dict['neuron_state'][gid]['I_t'] for gid in range(len(state_dict['neuron_state']))], dtype=np.float32)
    I_t_vector1 = I_t_list[:h_dim]
    I_t_vector2 = I_t_list[h_dim:]

    print("Run FC")

    FC_cpu(M_t, spike_per_ts, weight_per_spike, conn_mem_1_weight_transpose, conn_mem_2_weight_transpose, I_t_vector1, I_t_vector2, external_dim, h_dim, o_dim, num_thread, bin_width)
