import sys
import numpy as np
import cupy as cp

import time
import timeit

import configparser
from cupyx.profiler import benchmark
import argparse


np.random.seed(0)
debug = True

def SlayerFCSNN_gpu_realtime(M_t, bin_width, window, weight_matrix1, weight_matrix2,
                             weight_accum_matrix, neuron_states_refr, I_t_vector, v_t_vector,
                             decay_v_vector, g_t_vector,
                             decay_g_vector, threshold_vector, h_dim, num_neuron, sparse):

    # cpu to gpu memcpy
    # FIXME : loading the whole input matrix cannot be done
    # input_matrix = cp.asarray(input_matrix)

    weight_accum_matrix = cp.asarray(weight_accum_matrix)
    weight_matrix1 = cp.asarray(weight_matrix1)
    weight_matrix2 = cp.asarray(weight_matrix2)
    g_t_vector = cp.asarray(g_t_vector)
    I_t_vector = cp.asarray(I_t_vector)
    v_t_vector = cp.asarray(v_t_vector)
    decay_v_vector = cp.asarray(decay_v_vector)
    decay_g_vector = cp.asarray(decay_g_vector)
    threshold_vector = cp.asarray(threshold_vector)
    threshold_vector = threshold_vector[:h_dim]

    # save initial states for subsequent iteration use
    init_weight_accum_matrix = weight_accum_matrix.copy()
    init_g_t_vector = g_t_vector.copy()
    init_v_t_vector = v_t_vector.copy()

    # run SlayerFCSNN with cupy
    time_sum = 0
    num_iter = 1

    output_v_t = cp.zeros((M_t, num_neuron), dtype=cp.float32)

    if sparse:
        I_t_vector1 = cp.tile(I_t_vector[:h_dim], bin_width)
        I_t_vector2 = cp.tile(I_t_vector[h_dim:], bin_width)
    else:
        I_t_vector1 = I_t_vector[:h_dim].reshape(-1, 1)
        I_t_vector2 = I_t_vector[h_dim:].reshape(-1, 1)
    v_t_vector1 = v_t_vector[:h_dim]
    v_t_vector2 = v_t_vector[h_dim:]
    g_t_vector1 = g_t_vector[:h_dim]
    g_t_vector2 = g_t_vector[h_dim:]
    decay_v_vector1 = decay_v_vector[:h_dim]
    decay_v_vector2 = decay_v_vector[h_dim:]
    decay_g_vector1 = decay_g_vector[:h_dim]
    decay_g_vector2 = decay_g_vector[h_dim:]
    hidden_spike_matrix = cp.zeros((h_dim, bin_width), dtype=cp.float32)

    v_t_result = []

    if sparse:
        weight_matrix1 = cp.sparse.csr_matrix(weight_matrix1.transpose(1, 0, 2).reshape(-1, input_dim), dtype=cp.float32)
        weight_matrix2 = cp.sparse.csr_matrix(weight_matrix2.transpose(1, 0, 2).reshape(-1, input_dim), dtype=cp.float32)

    for iteration in range(num_iter):
        elapsed_time = 0

        # initialize state
        weight_accum_matrix1 = init_weight_accum_matrix[:h_dim]
        weight_accum_matrix2 = init_weight_accum_matrix[h_dim:]
        g_t_vector = init_g_t_vector.copy()
        v_t_vector = init_v_t_vector.copy()

        if sparse:
            weight_accum_matrix1 = weight_accum_matrix1.T.flatten()
            weight_accum_matrix2 = weight_accum_matrix2.T.flatten()

        for t in range(M_t):
            input_matrix = np.load('input_matrix.npy', allow_pickle=True, mmap_mode='r')
            if sparse:
                input_chunk = cp.asarray(input_matrix[t])
                tic=timeit.default_timer()
                input_vector = cp.sparse.csr_matrix(input_chunk, dtype=cp.float32)
                delta_weight1 = weight_matrix1.dot(input_vector)

                # accum weight
                timestep = t * bin_width

                window_idx = timestep % window

                for bin_idx in range(bin_width):
                    timestep_temp = timestep + bin_idx
                    window_idx_temp = timestep_temp % window
                    idx1_h = window_idx_temp * h_dim
                    idx2_h = (window - window_idx_temp) * h_dim
                    # if idx2_h > 0:
                    weight_accum_matrix1[idx1_h:] += delta_weight1[:idx2_h, bin_idx].todense().flatten()
                    # if idx1_h > 0:
                    weight_accum_matrix1[:idx1_h] += delta_weight1[idx2_h:, bin_idx].todense().flatten()

                weight_vector1 = weight_accum_matrix1[h_dim*window_idx:h_dim*(window_idx + bin_width)] + I_t_vector1 # I_t_vector will be broadcasted

                # pop weight_accum_matrix1
                weight_accum_matrix1[h_dim*window_idx:h_dim*(window_idx+bin_width)] = 0

                hidden_spike_matrix = cp.zeros((h_dim, bin_width), dtype=cp.float32) # this is to handle dimension mismatch
                for bin_idx in range(bin_width):
                    g_t_state1 = g_t_vector1 # NOTE : g_t_vector is for 1ms
                    if timestep > 0 or bin_idx > 0:
                        g_t_state1 += weight_vector1[h_dim*bin_idx:h_dim*(bin_idx+1)]
                        v_t_vector1 = v_t_vector1 * decay_v_vector1 + g_t_state1
                        hidden_vector = v_t_vector1 > threshold_vector
                        v_t_vector1[hidden_vector] = 0
                        hidden_spike_matrix[:, bin_idx] = hidden_vector
                    g_t_vector1 = decay_g_vector1 * g_t_state1

                # convert hidden_spike_matrix to CSR format
                hidden_spike_matrix = cp.sparse.csr_matrix(hidden_spike_matrix, dtype=cp.float32)

                # compute for layer 2 using hidden_spike_matrix
                delta_weight2 = weight_matrix2.dot(hidden_spike_matrix)

                # accum weight
                for bin_idx in range(bin_width):
                    if timestep > 0 or bin_idx > 0:
                        timestep_temp = timestep + bin_idx
                        window_idx_temp = timestep_temp % window
                        idx1_o = window_idx_temp * o_dim
                        idx2_o = (window - window_idx_temp) * o_dim

                        # if idx2_o > 0:
                        weight_accum_matrix2[idx1_o:] += delta_weight2[:idx2_o, bin_idx].todense().flatten()
                        # if idx1_o > 0:
                        weight_accum_matrix2[:idx1_o] += delta_weight2[idx2_o:, bin_idx].todense().flatten()

                # update g_t_state for output neurons
                # consider refractory period
                weight_vector2 = weight_accum_matrix2[o_dim*window_idx:o_dim*(window_idx+bin_width)] + I_t_vector2 # I_t_vector will be broadcasted
                # pop weight_accum_matrix2
                weight_accum_matrix2[o_dim*window_idx:o_dim*(window_idx+bin_width)] = 0
                # weight_vector2 shape (o_dim, bin_width)
                # weight_vector2 is precomputed
                # update v_t_vector2 and g_t_vector2 -> these are used to generate hidden neuron spike vector
                # use for loop
                for bin_idx in range(bin_width):
                    g_t_state2 = g_t_vector2
                    if timestep > 0 or bin_idx > 1:
                        g_t_state2 += weight_vector2[o_dim*bin_idx:o_dim*(bin_idx+1)]
                        v_t_vector2 = v_t_vector2 * decay_v_vector2 + g_t_state2
                    g_t_vector2 = decay_g_vector2 * g_t_state2
                    if debug:
                        pass
                        v_t_result.append(v_t_vector2)

                toc=timeit.default_timer()
                elapsed_time += toc - tic

            else:
                input_chunk = cp.asarray(input_matrix[t])
                tic=timeit.default_timer()

                # layer 1
                input_vector = input_chunk
                delta_weight1 = weight_matrix1.dot(input_vector)
                # delta_weight1 shape : (dst_neu, delay, bin_width)
                # accum weight
                timestep = t * bin_width

                window_idx = timestep % window

                for bin_idx in range(bin_width):
                    timestep_temp = timestep + bin_idx
                    window_idx_temp = timestep_temp % window
                    weight_accum_matrix1[:, window_idx_temp:] += delta_weight1[:, :window-window_idx_temp, bin_idx]
                    weight_accum_matrix1[:, :window_idx_temp] += delta_weight1[:, window-window_idx_temp:, bin_idx]

                # update g_t_state for hidden neurons
                # consider refractory period
                weight_vector1 = weight_accum_matrix1[:, window_idx:window_idx+bin_width] + I_t_vector1 # I_t_vector will be broadcasted
                # pop weight_accum_matrix1
                weight_accum_matrix1[:, window_idx:window_idx+bin_width] = 0
                # weight_vector1 shape (h_dim, bin_width)
                for bin_idx in range(bin_width):
                    g_t_state1 = g_t_vector1
                    if timestep > 0 or bin_idx > 0:
                        g_t_state1 += weight_vector1[:, bin_idx]
                        v_t_vector1 = v_t_vector1 * decay_v_vector1 + g_t_state1
                        hidden_vector = v_t_vector1 > threshold_vector
                        v_t_vector1[hidden_vector] = 0
                        hidden_spike_matrix[:, bin_idx] = hidden_vector
                    g_t_vector1 = decay_g_vector1 * g_t_state1

                # compute for layer 2 using hidden_spike_matrix
                delta_weight2 = weight_matrix2.dot(hidden_spike_matrix)

                # accum weight
                for bin_idx in range(bin_width):
                    if timestep > 0 or bin_idx > 0:
                        timestep_temp = timestep + bin_idx
                        window_idx_temp = timestep_temp % window

                        weight_accum_matrix2[:, window_idx_temp:] += delta_weight2[:, :window-window_idx_temp , bin_idx]
                        weight_accum_matrix2[:, :window_idx_temp ] += delta_weight2[:, window-window_idx_temp:, bin_idx]

                # update g_t_state for output neurons
                # consider refractory period
                weight_vector2 = weight_accum_matrix2[:, window_idx:window_idx+bin_width] + I_t_vector2 # I_t_vector will be broadcasted
                # pop weight_accum_matrix2
                weight_accum_matrix2[:, window_idx:window_idx+bin_width] = 0
                # weight_vector2 shape (o_dim, bin_width)
                # update v_t_vector2 and g_t_vector2 -> these are used to generate hidden neuron spike vector
                # use for loop
                for bin_idx in range(bin_width):
                    g_t_state2 = g_t_vector2
                    if timestep > 0 or bin_idx > 1:
                        g_t_state2 += weight_vector2[:, bin_idx]
                        v_t_vector2 = v_t_vector2 * decay_v_vector2 + g_t_state2
                    g_t_vector2 = decay_g_vector2 * g_t_state2
                    if debug:
                        v_t_result.append(v_t_vector2)

                toc=timeit.default_timer()
                elapsed_time += toc - tic

            del input_matrix

        time_sum += elapsed_time

    # for debugging - save output voltage
    result_file = open('output.txt', 'w')
    for v_t_line in v_t_result:
        result_file.write(str(v_t_line) + '\n')
    result_file.close()

    average = time_sum / (M_t*bin_width) * 1000 / num_iter
    print(f"Average per ts: {average} (ms)")

    return


if __name__ == '__main__':
    # Load simulation parameters
    properties = configparser.ConfigParser()
    properties.read('snn.params')

    dataset_dir = properties.get('general','dataset_dir')
    sim_duration = properties.getint('general','duration')
    sparse = properties.getboolean('general','gpu_sparse')

    window = properties.getint('general','window')
    bin_width = properties.getint('general','bin_width')

    assert (window > bin_width), "window size should be bigger than bin_width"
    assert (window > 256), "window size should be bigger than the maximum delay of neuron spikes"
    assert (window % bin_width == 0), "window size should be the multiple of bin_width"

    # NOTE : parse argument from the command
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_cell_num', type=int, dest='target_cell_num', default=-1, help='target_cell_num')
    args = parser.parse_args()

    target_cell_num = args.target_cell_num

    #sim_num_workers = properties.getint('general','num_workers')

    if target_cell_num < 0:
        target_cell_num = properties.getint('general','target_cell_num')

    hidden_dimension = properties.getint('general','h_dim')

    # 1. load external spike data
    external_stimulus = np.load(dataset_dir + 'external_stimulus.npy', allow_pickle=True)
    network_params = np.load(dataset_dir + 'network_parameter.npy', allow_pickle=True)

    # print(network_params)
    duration = network_params.item().get('simtime')

    if sim_duration > 0:
        assert(sim_duration % bin_width == 0) , "sim duration should be multiple of bin_width"
        duration = min(duration, sim_duration)

    input_dim = target_cell_num
    h_dim = hidden_dimension
    o_dim = 2
    M_t = duration

    M_t = duration // bin_width

    # extract stimulus
    input_matrix = np.zeros((M_t, input_dim, bin_width), dtype=np.float32)
    # print(input_matrix)
    # print("shape", input_matrix.shape)

    for spike in external_stimulus:
        timestep = int(spike[0])
        neuron_id = int(spike[1])
        spike_value = spike[2]
        if timestep >= duration:
            break
        M_t_prime = timestep // bin_width
        bin_idx = timestep % bin_width
        input_matrix[M_t_prime, neuron_id, bin_idx] = 1

    # # NOTE : save input matrix as npy file and (partially)load in the FC gpu process
    np.save("input_matrix", input_matrix)

    external_dim = target_cell_num
    external_offset = h_dim + o_dim
    num_neuron = h_dim + o_dim

    print("=====================================")
    print("duration:", duration)
    print("M_t:", M_t)
    print("target cell num:", target_cell_num)
    print("window:", window, "(bins)")
    print("h_dim:", h_dim)
    print("o_dim:", o_dim)
    print("bin_width", bin_width)
    print("gpu sparse :", sparse)
    print("=====================================")
    cp.cuda.Device(2).use()

    # TODO : load connection data in appropriate form for CuPy
    conn_dict = np.load(dataset_dir + 'connection.npy', allow_pickle=True)

    # weight_matrix1 for layer 1 -> (dst neuron, delay, src neuron)
    weight_matrix1 = np.zeros((h_dim, window, input_dim), dtype=np.float32)
    # weight_matrix2 for layer 2 -> (dst neuron, delay, src neuron)
    weight_matrix2 = np.zeros((o_dim, window, h_dim), dtype=np.float32)

    conn_mem_1_temp = conn_dict[1][0] # bci_neuron to neuron
    conn_mem_1_len = len(conn_mem_1_temp)
    conn_mem_2_temp = conn_dict[0][0] # neuron to neuron
    conn_mem_2_len = len(conn_mem_2_temp)

    # 1. bci neuron to neuron connection
    for conn_idx in range(len(conn_mem_1_temp)):
        conn = conn_mem_1_temp[conn_idx]
        src = int(conn[0])
        dst = int(conn[1])
        has_entry = bool(conn[2])
        entry = conn[3]

        weight = entry[0][1]
        delay = entry[1][1]

        assert(delay == 1) # bci neuron to neuron delay is always 1
        weight_matrix1[dst, delay, src] = weight

    # 2. neuron to neuron connection
    for conn_idx in range(len(conn_mem_2_temp)):
        conn = conn_mem_2_temp[conn_idx]
        src = int(conn[0])
        dst = int(conn[1])
        has_entry = bool(conn[2])
        entry = conn[3]

        weight = entry[0][1]
        delay = entry[1][1]

        dst = dst - h_dim

        weight_matrix2[dst, delay, src] = weight
        # print(delay)

    print("Build connection done")

    # 3. required states & parameters for SlayerFCSNN
    print("Load states and parameters")
    state_dict = np.load(dataset_dir + 'initial_states.npy', allow_pickle=True).item()

    refr_list = np.array([state_dict['neuron_state'][gid]['refr'] for gid in range(len(state_dict['neuron_state']))], dtype=np.int32)
    I_t_list = np.array([state_dict['neuron_state'][gid]['I_t'] for gid in range(len(state_dict['neuron_state']))], dtype=np.single)
    v_t_list = np.array([state_dict['neuron_state'][gid]['v_t'] for gid in range(len(state_dict['neuron_state']))], dtype=np.single)
    decay_v_list = np.array([state_dict['neuron_state'][gid]['decay_v'] for gid in range(len(state_dict['neuron_state']))], dtype=np.single)
    g_t_list = np.array([state_dict['neuron_state'][gid]['g_t'] for gid in range(len(state_dict['neuron_state']))], dtype=np.single)
    decay_g_list = np.array([state_dict['neuron_state'][gid]['decay_g'] for gid in range(len(state_dict['neuron_state']))], dtype=np.single)
    threshold_list = np.array([state_dict['neuron_state'][gid]['threshold'] for gid in range(len(state_dict['neuron_state']))], dtype=np.single)

    # weight accum
    weight_accum = np.zeros((num_neuron, window), dtype=np.float32)

    print("Run SlayerFCSNN")
    SlayerFCSNN_gpu_realtime(M_t, bin_width, window, weight_matrix1, weight_matrix2,
                             weight_accum, refr_list, I_t_list, v_t_list, decay_v_list, g_t_list,
                             decay_g_list, threshold_list, h_dim, num_neuron, sparse)
