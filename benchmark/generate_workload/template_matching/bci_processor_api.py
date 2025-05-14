import os
import sys

import numpy as np
import math

#
network_params = {}
neu_states = None

simtime_dat = None
dt_dat = None

unit_params = {}
conn_list = {}

def set_seed(seed: int):
    np.random.seed(seed)


def init_simulation(simtime: object,
                    dt: object,
                    N_n: int,
                    N_tm: int,
                    N_t: int,
                    result_path):
    global simtime_dat
    global dt_dat
    global network_params
    global result_dir

    result_dir = result_path

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)


    simtime_dat = simtime
    dt_dat = dt
    network_params["simtime"] = int(simtime / dt)
    network_params["dt"] = dt

    network_params["num_internal"] = N_n + N_tm
    network_params["num_unit"] = [N_n, N_tm, N_n]
    network_params["num_unit_types"] = 3
    network_params["unit_types"] = ["neuron", "template", "bci"]
    network_params["consts"] = {'n_t': N_t}
    print("Initialized Simulation")
    print("cell num:", N_n)
    sys.stdout.flush()


def create_external_stimulus(num_external: int,
                             external_stim: list):

    global network_params
    global external_stimulus

    network_params["num_external"] = num_external # 0
    external_stimulus = external_stim # empty array

    print("Created External Stimulus")
    sys.stdout.flush()


def create_units(constants: list):

    global unit_params

    unit_params["temp_consts"] = constants
    # print(unit_params)

    print("Created Internal Units (Templates)")
    sys.stdout.flush()


def create_connections(num_neurons: int,
                       N_t: list,
                       temp_neurons: list,
                       templates: list,
                       bin_width: int):

    global conn_list
    global mapping

    N_tm = len(templates)
    N_n = num_neurons

    num_types = 3
    num_unit = [N_n, N_tm, N_n]

    # print(temp_neurons)
    # print(len(temp_neurons), N_n, N_tm)

    # temp_to_neu = {}
    neu_to_temp = {}

    for neu in range(N_n):
        neu_to_temp[neu] = []
        for tm in range(N_tm):
            # temp_to_neu[tm] = list(temp_neurons[tm])
            if neu in temp_neurons[tm]:
                neu_to_temp[neu].append(tm)

    conn_list = [[[] for _ in range(num_types)] for _ in range(num_types)]

    offset = [0]
    for dat in num_unit:
        offset += [offset[-1] + dat]

    offset = offset[:-1]

    mapping = []

    for t1 in range(num_types):
        for t2 in range(num_types):
            t1_to_t2_conn_list = []
            if (t1 == 0 and t2 == 1): # neu to temp conn
                for i in neu_to_temp.keys():
                    for j in list(neu_to_temp[i]):
                        # Set the TM bit precision
                        load = [('template', templates[j][i], int(np.log2((bin_width - 1) // 5 + 1) + 1))]
                        t1_to_t2_conn_list.append((i, j, True, load))
                conn_list[t1][t2] = t1_to_t2_conn_list
            if (t1 == 2 and t2 == 0): # bci to neu conn
                for i in range(num_neurons):
                    t1_to_t2_conn_list.append((i, i, False, None))
                conn_list[t1][t2] = t1_to_t2_conn_list

    for src_type in range(num_types):
        for dst_type in range(num_types):
            connection = conn_list[src_type][dst_type]
            src_offset = offset[src_type]
            dst_offset = offset[dst_type]
            for conn in connection:
                src, dst, en, dat = conn
                mapping.append((src + src_offset, dst + dst_offset))

    # print("neu_to_temp")
    # print(neu_to_temp)
    # print("conn_list")
    # print(conn_list)
    # print("mapping", mapping)

    print("Created Connections")
    sys.stdout.flush()


def end_simulation():

    global network_params
    global neu_states
    global recording_params

    global simtime_dat
    global dt_dat

    global unit_params
    global conn_list

    # check if the network has been properly initialized
    assert(network_params)
    #assert(neu_states)
    assert(unit_params)
    assert(conn_list)

    # set bit precision for hist mem
    unit_params['init_prec'] = {'R2': 16, 'R3': 16, 'P1': 16}

    np.save(result_dir + "/external_stimulus.npy", external_stimulus) # spike data
    np.save(result_dir + "/network_parameter.npy", network_params) # N_t, N_n, N_tm
    np.save(result_dir + "/initial_states.npy", unit_params) # neuron spike data, template constants (C1, C2, C3)
    conn_list = np.array(conn_list, dtype=object)
    np.save(result_dir + "/connection.npy", conn_list) # connection data
    np.save(result_dir + "/mapping.npy", mapping) # mapping file

    print("Generated Required Metadata")
    sys.stdout.flush()

    network_params = {}
    neu_states = None

    simtime_dat = None
    dt_dat = None

    unit_params = {}
    conn_list = []

