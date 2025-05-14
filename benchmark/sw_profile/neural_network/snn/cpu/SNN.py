import sys, os
import numpy as np

import time
import timeit

import cProfile
import yappi

import configparser
import inference
import argparse

np.random.seed(0)
debug = True

# NOTE : epuivalent to process_one_epoch
def SlayerFCSNN_cpu(M_t, spike_per_ts,
                             conn_mem_1_delay, conn_mem_1_weight,
                             conn_mem_2_delay, conn_mem_2_weight,
                             I_t_vector1, I_t_vector2, decay_v_vector1 ,decay_v_vector2,
                             g_t_vector1, g_t_vector2, decay_g_vector1, decay_g_vector2,
                             threshold_list, input_dim, h_dim, o_dim, window, num_neuron, num_thread):

    time_sum = 0
    num_iter = 1

    running_loss = 0
    output_v_t = [None for _ in range(M_t)]

    yappi.set_clock_type("wall")
    yappi.start()
    for iteration in range(num_iter):
        elapsed_time = 0

        hidden_spiked_list = np.zeros(h_dim, dtype=np.int32)
        hidden_vector = np.zeros(h_dim, dtype=np.float32)
        output_vector = np.zeros(o_dim, dtype=np.float32)

        weight_accum1 = np.zeros((h_dim, window), dtype=np.float32)
        weight_accum2 = np.zeros((o_dim, window), dtype=np.float32)

        for t in range(M_t): # iterate over timesteps
            tic=timeit.default_timer()
            # get the list of neurons spiked at the given ts
            spike_list = np.array(spike_per_ts[t], dtype=np.int32)
            num_spikes = spike_list.shape[0]

            inference.SlayerFCSNN_cpu_inner(spike_list, conn_mem_1_delay, conn_mem_1_weight, conn_mem_2_delay, conn_mem_2_weight,
                    weight_accum1, weight_accum2, I_t_vector1, I_t_vector2, hidden_vector, output_vector,
                    decay_v_vector1, decay_v_vector2, g_t_vector1, g_t_vector2, decay_g_vector1, decay_g_vector2,
                    threshold, hidden_spiked_list, input_dim,  h_dim, o_dim, t, window, num_spikes, num_thread)

            toc=timeit.default_timer()
            elapsed_time += toc - tic

            # save current output_vector as output at this timestep
            output_v_t[t] = output_vector.copy()

        time_sum += elapsed_time
        print(f"Elapsed: {elapsed_time} (s)")
    yappi.stop()

    threads = yappi.get_thread_stats()
    for thread in threads:
        print("Function stats for (%s) (%d)"%(thread.name, thread.id))
        yappi.get_func_stats(ctx_id=thread.id).print_all()

    print(f"Num iter: {num_iter}")
    print(f"Average per iter: {time_sum / num_iter} (s)")

    # for debugging
    result_file = open('output.txt', 'w')
    for v_t_line in output_v_t:
        result_file.write(str(list(v_t_line)) + '\n')
    result_file.close()

    average = time_sum / M_t * 1000 / num_iter
    print(f"Average per ts: {average} (ms)")

#@profile
if __name__ == '__main__':
    # Load simulation parameters
    properties = configparser.ConfigParser()
    properties.read('snn.params')

    dataset_dir = properties.get('general','dataset_dir')
    sim_duration = properties.getint('general','duration')

    window = properties.getint('general','window')

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

    hidden_dimension = properties.getint('general','h_dim')

    # 1. load external spike data
    external_stimulus = np.load(dataset_dir+'external_stimulus.npy', allow_pickle=True)
    network_params = np.load(dataset_dir+'network_parameter.npy', allow_pickle=True)

    print(network_params)
    duration = network_params.item().get('simtime')

    if sim_duration > 0:
        duration = min(duration, sim_duration)

    input_dim = target_cell_num
    h_dim = hidden_dimension
    o_dim = 2
    M_t = duration

    external_dim = target_cell_num
    external_offset = h_dim + o_dim
    num_neuron = h_dim + o_dim

    print("=====================================")
    print("M_t:", M_t)
    print("target cell num:", target_cell_num)
    print("window:", window, "(bins)")
    print("h_dim:", h_dim)
    print("o_dim:", o_dim)
    print(f"cpus : {num_thread}")
    print("=====================================")

    # convert this to spike idx per ts
    print('Spike Compression Start')
    spike_per_ts = [[] for _ in range(M_t)]
    # weight_per_spike = [[] for _ in range(M_t)]

    for spike in external_stimulus:
        timestep = int(spike[0])
        neuron_id = int(spike[1])
        spike_value = spike[2]
        if timestep >= M_t:
            break
        if spike_value != 0:
            spike_per_ts[timestep].append(neuron_id)
            # weight_per_spike[timestep].append(spike_value)
    print('Spike Compression Done')

    # 2. load connection data
    conn_dict = np.load(dataset_dir+'connection.npy', allow_pickle=True)

    conn_mem_1_temp = conn_dict[1][0] # bci_neuron to neuron
    conn_mem_1_len = len(conn_mem_1_temp)
    conn_mem_2_temp = conn_dict[0][0] # neuron to neuron
    conn_mem_2_len = len(conn_mem_2_temp)

    conn_mem_1_delay = np.zeros(input_dim * h_dim, dtype = np.int32)
    conn_mem_1_weight = np.zeros(input_dim * h_dim, dtype = np.single)
    conn_mem_2_delay = np.zeros(h_dim * o_dim, dtype = np.int32)
    conn_mem_2_weight = np.zeros(h_dim * o_dim, dtype = np.single)

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

        conn_mem_1_delay[dst * input_dim + src] = delay
        conn_mem_1_weight[dst * input_dim + src] = weight

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

        conn_mem_2_delay[dst * h_dim + src] = delay
        conn_mem_2_weight[dst * h_dim + src] = weight

    print("Build connection done")

    # 3. required states & parameters for SlayerFCSNN
    print("Load states and parameters")
    state_dict = np.load(dataset_dir+'initial_states.npy', allow_pickle=True).item()

    refr_list = np.array([state_dict['neuron_state'][gid]['refr'] for gid in range(len(state_dict['neuron_state']))], dtype=np.int32)
    I_t_list = np.array([state_dict['neuron_state'][gid]['I_t'] for gid in range(len(state_dict['neuron_state']))], dtype=np.single)
    v_t_list = np.array([state_dict['neuron_state'][gid]['v_t'] for gid in range(len(state_dict['neuron_state']))], dtype=np.single)
    decay_v_list = np.array([state_dict['neuron_state'][gid]['decay_v'] for gid in range(len(state_dict['neuron_state']))], dtype=np.single)
    g_t_list = np.array([state_dict['neuron_state'][gid]['g_t'] for gid in range(len(state_dict['neuron_state']))], dtype=np.single)
    decay_g_list = np.array([state_dict['neuron_state'][gid]['decay_g'] for gid in range(len(state_dict['neuron_state']))], dtype=np.single)
    threshold_list = np.array([state_dict['neuron_state'][gid]['threshold'] for gid in range(len(state_dict['neuron_state']))], dtype=np.single)

    I_t_vector1 = I_t_list[:h_dim]
    I_t_vector2 = I_t_list[h_dim:]

    decay_v_vector1 = decay_v_list[:h_dim]
    decay_v_vector2 = decay_v_list[h_dim:]

    g_t_vector1 = g_t_list[:h_dim]
    g_t_vector2 = g_t_list[h_dim:]

    decay_g_vector1 = decay_g_list[:h_dim]
    decay_g_vector2 = decay_g_list[h_dim:]

    threshold = threshold_list[:h_dim]

    print("Run SlayerFCSNN")

    SlayerFCSNN_cpu(M_t, spike_per_ts,
                    conn_mem_1_delay, conn_mem_1_weight,
                    conn_mem_2_delay, conn_mem_2_weight,
                    I_t_vector1, I_t_vector2, decay_v_vector1, decay_v_vector2,
                    g_t_vector1, g_t_vector2, decay_g_vector1, decay_g_vector2,
                    threshold, input_dim, h_dim, o_dim, window, num_neuron, num_thread)
