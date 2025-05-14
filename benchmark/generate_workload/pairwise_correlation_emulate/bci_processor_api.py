import os 
import sys

import numpy as np
import math

#
network_params = {}
neu_states = None
recording_params = {}

window_dat = None

template_params = {}
conn_list = {}

initial_params = {}

def set_seed(seed: int):
    np.random.seed(seed)


def init_simulation(simtime: object,
                    dt: object,
                    N: int,
                    window: int,
                    corr_period: int,
                    result_path):
    global network_params
    global window_dat
    global result_dir
    
    result_dir = result_path

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    window_dat = window + 1
    network_params['simtime'] = int(simtime / dt)
    network_params['dt'] = dt
    network_params['num_internal'] = None
    network_params['num_unit'] = [N, N]
    network_params['num_unit_types'] = 2
    network_params['unit_types'] = ['neuron', 'bci']
    network_params['consts'] = {'window': window, 'corr_period': corr_period, 'conn_to_src' : None}

    print('Initialized Simulation')
    sys.stdout.flush()

def create_external_stimulus(num_external: int,
                             external_stim: list):

    global network_params
    global external_stimulus # save recording data here

    network_params['num_external'] = num_external
    external_stimulus = external_stim

    print('Created Electrodes')
    sys.stdout.flush()

def create_connections(num_neurons: int,
                       connection: list,
                       conn_dict : dict):

    global network_params
    global conn_list
    global mapping
    global window_dat

    network_params['num_internal'] = network_params['num_unit'][0]

    num_types = network_params['num_unit_types']
    num_unit = network_params['num_unit']

    conn_list = [[[] for _ in range(num_types)] for _ in range(num_types)]

    offset = [0]
    for dat in num_unit:
        offset += [offset[-1] + dat]
    offset = offset[:-1]
    mapping = []

    for t1 in range(num_types):
        for t2 in range(num_types):
            t1_name = network_params['unit_types'][t1]
            t2_name = network_params['unit_types'][t2]

            t1_to_t2_conn_list = []
            #if (t1_name == 'neuron' and t2_name == 'partial_connection'): # neu to partial temp conn
            #    for i, j in connection:
            #        for t in range(window_dat):
            #            conn_idx = conn_dict[(i, j)]
            #            t1_to_t2_conn_list.append((i, conn_idx * window_dat + t, False, None))
            #    conn_list[t1][t2] = t1_to_t2_conn_list
            if (t1_name == 'neuron' and t2_name == 'neuron'): # neu to partial temp conn
                for i, j in connection:
                    for t in range(window_dat):
                        conn_idx = conn_dict[(i, j)]
                        load = [('partial_correlation', 0, 10), ('delay', -t, 10)]
                        t1_to_t2_conn_list.append((i, j, True, load))
                conn_list[t1][t2] = t1_to_t2_conn_list
            if (t1_name == 'bci' and t2_name == 'neuron'): # neu to partial temp conn
                for i in range(num_neurons):
                    t1_to_t2_conn_list.append((i, i, False, None))
                conn_list[t1][t2] = t1_to_t2_conn_list

    conn_to_src = {}
    for i, j in connection:
        for t in range(window_dat):
            conn_idx = conn_dict[(i, j)]
            conn_to_src[conn_idx * window_dat + t + network_params['num_unit'][0]] = i
    network_params['consts']['conn_to_src'] = conn_to_src

    for src_type in range(num_types):
        for dst_type in range(num_types):
            connection = conn_list[src_type][dst_type]
            src_offset = offset[src_type]
            dst_offset = offset[dst_type]
            for conn in connection:
                src, dst, en, dat = conn
                mapping.append((src + src_offset, dst + dst_offset))

    print('Created Connections')
    print('')
    sys.stdout.flush()


def end_simulation():

    global network_params
    global neu_states

    global window_dat
    
    global initial_params
    global conn_list
    
    global result_dir

    # check if the network has been properly initialized
    assert(network_params)
    assert(conn_list)

    # set bit precision for hist mem
    initial_params['init_prec'] = {'history': 10}

    np.save(result_dir + '/external_stimulus.npy', external_stimulus) # spike data
    np.save(result_dir + '/network_parameter.npy', network_params)
    np.save(result_dir + '/initial_states.npy', initial_params) 
    conn_list = np.array(conn_list, dtype=object)
    np.save(result_dir + '/connection.npy', conn_list)
    np.save(result_dir + '/mapping.npy', mapping) # mapping file

    print('Generated Required Metadata')
    sys.stdout.flush()

    network_params = {}
    neu_states = None
    
    window_dat = None
    
    conn_list = []
