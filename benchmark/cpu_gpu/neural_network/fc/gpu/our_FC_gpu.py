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

def FC_gpu_realtime(M_t, weight_matrix1, weight_matrix2, I_t_vector1, I_t_vector2, i_dim, h_dim, o_dim, bin_width, sparse):

    # cpu to gpu memcpy
    # spike_per_ts = cp.asarray(spike_per_ts, dtype=cp.int32)
    # NOTE : loading too much memory to the GPU is not feasible
    # input_matrix = cp.asarray(input_matrix, dtype=cp.float32)
    weight_matrix1 = cp.asarray(weight_matrix1.T, dtype=cp.float32)
    weight_matrix2 = cp.asarray(weight_matrix2.T, dtype=cp.float32)
    I_t_vector1 = cp.asarray(I_t_vector1, dtype=cp.float32)
    I_t_vector2 = cp.asarray(I_t_vector2, dtype=cp.float32)

    output_matrix = cp.zeros((M_t, o_dim), dtype=cp.float32)

    if sparse:
        # input_matrix = cp.sparse.csr_matrix(input_matrix)
        weight_matrix1 = cp.sparse.csr_matrix(weight_matrix1, dtype=cp.float32)
        weight_matrix2 = cp.sparse.csr_matrix(weight_matrix2, dtype=cp.float32)
        I_t_vector1 = cp.sparse.csr_matrix(I_t_vector1, dtype=cp.float32)
        I_t_vector2 = cp.sparse.csr_matrix(I_t_vector2, dtype=cp.float32)


    # run FC with cupy
    time_sum = 0
    num_iter = 10

    for iteration in range(num_iter):
        # NOTE : input chunk will contain single timestep data
        input_matrix = np.load('input_matrix.npy', allow_pickle=True, mmap_mode='r')
        input_chunk = cp.asarray(input_matrix[0], dtype=cp.float32)
        spike_buf = cp.zeros((bin_width, i_dim), dtype=cp.float32)
        spike_sum = cp.zeros((i_dim,), dtype=cp.float32)

        tic=timeit.default_timer()
        # dynamic binning
        spike_buf[0] = input_chunk
        spike_sum += input_chunk
        if sparse:
            input_vector = cp.sparse.csr_matrix(spike_sum, dtype=cp.float32)
        else:
            input_vector = spike_sum
        hidden_vector = input_vector.dot(weight_matrix1) + I_t_vector1

        toc=timeit.default_timer()

        time_sum += toc - tic
        # free memory
        del input_matrix

        for t in range(1, M_t): # iterate over 600ms
            input_matrix = np.load('input_matrix.npy', allow_pickle=True, mmap_mode='r')
            input_chunk = cp.asarray(input_matrix[t], dtype=cp.float32)
            spike_sum -= spike_buf[t%bin_width]
            spike_buf[t%bin_width] = input_chunk
            spike_sum += input_chunk

            tic=timeit.default_timer()

            if sparse:
                input_vector = cp.sparse.csr_matrix(spike_sum, dtype=cp.float32)
            else:
                input_vector = spike_sum

            # Layer 2 (hidden -> output) with ReLU

            if sparse:
                output_vector = hidden_vector.maximum(0).dot(weight_matrix2) + I_t_vector2
            else:
                output_vector = cp.maximum(hidden_vector, 0).dot(weight_matrix2) + I_t_vector2

            # Layer 1 (input -> hidden)
            hidden_vector = input_vector.dot(weight_matrix1) + I_t_vector1

            toc=timeit.default_timer()
            time_sum += toc - tic

            # for debugging -> save output_vector to output_matrix
            if iteration == 0:
                if sparse:
                    output_matrix[t] = output_vector.todense()
                else:
                    output_matrix[t] = output_vector

            del input_matrix

        cp.cuda.Device().synchronize()

    # for debugging
    result_file = open('output.txt', 'w')
    for v_t_line in output_matrix:
        result_file.write(str(v_t_line) + '\n')
    result_file.close()

    average = time_sum / M_t * 1000 / num_iter
    print(f"Average per iter: {time_sum/num_iter} (s)")
    print(f"Average per ts: {average} (ms)")

    return


if __name__ == '__main__':
    # Load simulation parameters
    properties = configparser.ConfigParser()
    properties.read('fc.params')

    dataset_dir = properties.get('general','dataset_dir')
    sim_duration = properties.getint('general','duration')

    sparse = properties.getboolean('general','gpu_sparse')

    # NOTE : parse argument from the command
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_cell_num', type=int, dest='target_cell_num', default=-1, help='target_cell_num')
    args = parser.parse_args()

    target_cell_num = args.target_cell_num

    #sim_num_workers = properties.getint('general','num_workers')

    if target_cell_num < 0:
        target_cell_num = properties.getint('general','target_cell_num')

    # 1. load external spike data
    external_stimulus = np.load(dataset_dir + 'external_stimulus.npy', allow_pickle=True)
    network_params = np.load(dataset_dir + 'network_parameter.npy', allow_pickle=True)

    hidden_dimension = properties.getint('general', 'h_dim')

    bin_width = properties.getint('general','bin_width')

    # print(network_params)
    duration = network_params.item().get('simtime')

    if sim_duration > 0:
        duration = min(duration, sim_duration)

    h_dim = hidden_dimension
    o_dim = 2
    M_t = duration

    # extract stimulus
    input_matrix = np.zeros((M_t, target_cell_num), dtype=np.float32)
    # print(input_matrix)
    # print("shape", input_matrix.shape)

    for spike in external_stimulus:
        timestep = int(spike[0])
        neuron_id = int(spike[1])
        spike_value = spike[2]
        if timestep >= M_t:
            break
        input_matrix[timestep, neuron_id] = spike_value

    # # NOTE : save input matrix as npy file and (partially)load in the FC gpu process
    np.save("input_matrix", input_matrix)

    external_dim = target_cell_num
    external_offset = h_dim + o_dim
    num_neuron = h_dim + o_dim

    print("=====================================")
    print("M_t:", M_t)
    print("target cell num:", target_cell_num)
    print("h_dim:", h_dim)
    print("o_dim:", o_dim)
    print("gpu sparse :", sparse)
    print("=====================================")

    # 2. load connection data
    conn_dict = np.load(dataset_dir + 'connection.npy', allow_pickle=True)

    # Layer 1
    conn_mem_1_temp = conn_dict[1][0] # bci_neuron to neuron
    conn_mem_1_len = len(conn_mem_1_temp)
    # Layer 2
    conn_mem_2_temp = conn_dict[0][0] # neuron to neuron
    conn_mem_2_len = len(conn_mem_2_temp)

    # NOTE : we need to build weight matrix using connection information

    weight_matrix1 = np.zeros((h_dim, target_cell_num), dtype=np.single)
    for conn_idx in range(len(conn_mem_1_temp)):
        conn = conn_mem_1_temp[conn_idx]
        src = int(conn[0])
        dst = int(conn[1])
        has_entry = bool(conn[2])
        entry = conn[3]

        weight = entry[0][1]
        delay = entry[1][1]

        weight_matrix1[dst][src] = weight
        # delay is 1 for FC layer

    weight_matrix2 = np.zeros((o_dim, h_dim), dtype=np.single)
    for conn_idx in range(len(conn_mem_2_temp)):
        conn = conn_mem_2_temp[conn_idx]
        src = int(conn[0])
        dst = int(conn[1])
        has_entry = bool(conn[2])
        entry = conn[3]

        weight = entry[0][1]
        delay = entry[1][1]

        dst = dst - h_dim # translate output neuron id to local id (e.g. 200, 201 -> 0, 1)

        weight_matrix2[dst][src] = weight
        # delay is 1 for FC layer

    assert(delay == 1) # delay is 1 for FC layer

    print("Build connection done")

    # 3. required states & parameters for SlayerFCSNN
    print("Load states and parameters")
    state_dict = np.load(dataset_dir + 'initial_states.npy', allow_pickle=True).item()

    # refr_list = np.array([state_dict['neuron_state'][gid]['refr'] for gid in range(len(state_dict['neuron_state']))], dtype=np.int32)
    I_t_list = np.array([state_dict['neuron_state'][gid]['I_t'] for gid in range(len(state_dict['neuron_state']))], dtype=np.single)

    # NOTE
    # we need all the parameters to be matrix form
    I_t_vector1 = I_t_list[:h_dim] # length of target cell num
    I_t_vector2 = I_t_list[h_dim:] # 2

    # print("weight_matrix1", weight_matrix1)
    # print("weight_matrix1 shape", weight_matrix1.shape)

    print("Run FC")
    FC_gpu_realtime(M_t, weight_matrix1, weight_matrix2, I_t_vector1, I_t_vector2, external_dim, h_dim, o_dim, bin_width, sparse)
