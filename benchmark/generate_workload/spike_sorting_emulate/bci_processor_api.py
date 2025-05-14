import os
import sys

import numpy as np
import math

import networkx as nx
from elephant.spike_train_generation import homogeneous_poisson_process

#
network_params = {}
recording_params = {}

simtime_dat = None
dt_dat = None

template_params = {}
conn_list = {}

initial_params = {}

def set_seed(seed: int):
    np.random.seed(seed)


def init_simulation(simtime: object,
                    dt: object,
                    N_e: int,
                    N_tm: int,
                    p_t: int,
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
    network_params["timestep_per_sec"] = int(1000 / dt)
    network_params["dt"] = dt
    network_params["num_internal"] = N_e + N_tm
    network_params["num_unit"] = [N_e, N_tm, N_e]
    network_params["num_unit_types"] = len(network_params["num_unit"])
    network_params["unit_types"] = ["electrode", "template", "bci"]
    network_params["consts"] = {'n_scalar' : N_e * N_t,
                                'n_t': N_t,
                                'template_shift': N_t // 2}


    print("Initialized Simulation")
    sys.stdout.flush()

def create_external_stimulus(num_external: int,
                             external_stim: list):

    global network_params
    global external_stimulus # save recording data here

    network_params["num_external"] = num_external
    external_stimulus = external_stim

    print("Created Electrodes")
    sys.stdout.flush()


def create_initial_states(norm_templates, amp_limits, thresholds, n_tm, n_scalar):

    global initial_params

    sub_norm_templates = n_scalar * norm_templates[:n_tm]
    min_scalar_products = amp_limits[:,0][:,np.newaxis]
    max_scalar_products = amp_limits[:,1][:,np.newaxis]
    min_sps = min_scalar_products * sub_norm_templates[:, np.newaxis]
    max_sps = max_scalar_products * sub_norm_templates[:, np.newaxis]

    sps = [(min_sps[i],max_sps[i]) for i in range(len(min_sps))]

    initial_params["norm_templates"] = norm_templates
    initial_params["thresholds"] = thresholds
    initial_params["sps"] = sps

    print("Created Initial Params")
    sys.stdout.flush()


def create_connections(temp_shape: list, templates: list):

    global network_params
    global conn_list
    global mapping

    n_e, n_t, n_tm = temp_shape
    idx = 0

    n_tm = n_tm // 2 # consider only primary template

    num_elec = n_e
    num_unit = network_params["num_unit"]
    num_types = network_params["num_unit_types"]

    templates = templates[:n_tm]
    templates_memory = [[None for _ in range(n_e)] for _ in range(n_tm)]
    for i in range(n_tm):
        for j in range(n_e):
            templates_memory[i][j] = templates[i][n_t*j:n_t*(j+1)]

    temp_to_elec = {}
    elec_to_temp = {}

    temp_count = 0
    for tm in range(n_tm):
        temp_to_elec[tm] = []
        for elec in range(n_e):
            if np.any(templates[tm][n_t*elec:n_t*(elec+1)]):
                temp_to_elec[tm].append(elec)
                temp_count += 1

    elec_count = 0
    for elec in range(n_e):
        elec_to_temp[elec] = []
        for tm in range(n_tm):
            if np.any(templates[tm][n_t*elec:n_t*(elec+1)]):
                elec_to_temp[elec].append(tm)
                elec_count += 1
    # print("Average # of temp/elec: {}, elec/temp: {}".format(float(temp_count / n_tm), float(elec_count / n_e)))

    conn_list = [[[] for _ in range(num_types)] for _ in range(num_types)]

    offset = [0]
    for dat in num_unit:
        offset += [offset[-1] + dat]

    offset = offset[:-1]

    mapping = []

    for t1 in range(num_types):
        for t2 in range(num_types):
            t1_name = network_params["unit_types"][t1]
            t2_name = network_params["unit_types"][t2]

            t1_to_t2_conn_list = []
            # For partial accumulation + peak detection
            if (t1_name == "electrode" and t2_name == "template"): # neu to partial temp conn
                for i in elec_to_temp.keys():
                    for j in list(elec_to_temp[i]):
                        t1_to_t2_conn_list.append((i, j, False, None))
                conn_list[t1][t2] = t1_to_t2_conn_list
            if (t1_name == "template" and t2_name == "electrode"): # neu to partial temp conn
                flattened_idx = 0
                for i in temp_to_elec.keys():
                    for j in list(temp_to_elec[i]):
                        for t in range(n_t):
                            load = [('template', templates_memory[i][j][t], 10), ('delay', t + 1 - n_t, int(np.log2(n_t)) + 1)]
                            t1_to_t2_conn_list.append((i, j, True, load))
                        flattened_idx += 1
                conn_list[t1][t2] = t1_to_t2_conn_list
            if (t1_name == "bci" and t2_name == "electrode"): # neu to partial temp conn
                for i in range(num_elec):
                    t1_to_t2_conn_list.append((i, i, False, None))
                conn_list[t1][t2] = t1_to_t2_conn_list

    for src_type in range(num_types):
        for dst_type in range(num_types):
            connection = conn_list[src_type][dst_type]
            src_offset = offset[src_type]
            dst_offset = offset[dst_type]
            for conn in connection:
                src, dst, en, dat = conn
                if src_type < num_types - 1 and dst_type < num_types - 1:
                    mapping.append((src + src_offset, dst + dst_offset))

#    print("temp_to_elec")
#    print(temp_to_elec)
#    print("elec_to_temp")
#    print(elec_to_temp)
#    print("conn_list")
#    print(conn_list)
#    print("mapping", mapping)

    print("Created Connections")
    sys.stdout.flush()


def end_simulation():

    global network_params

    global simtime_dat
    global dt_dat

    global initial_params
    global conn_list

    global external_stimulus

    global mapping

    # check if the network has been properly initialized
    assert(network_params)
    assert(conn_list)

    # set bit precision for hist mem
    initial_params['init_prec'] = {'bci': 10}

    np.save(result_dir + "/external_stimulus.npy", external_stimulus) # bci recording data
    np.save(result_dir + "/network_parameter.npy", network_params) # N_t, N_e, N_tm
    np.save(result_dir + "/initial_states.npy", initial_params) # norm_templates, (amp_limits), thresholds, min_sps, max_sps
    conn_list = np.array(conn_list, dtype=object)
    np.save(result_dir + "/connection.npy", conn_list) # connection data
    np.save(result_dir + "/mapping.npy", mapping) # mapping file

    print("Generated Required Metadata")
    sys.stdout.flush()

    network_params = {}

    simtime_dat = None
    dt_dat = None

    initial_params = {}
    conn_list = []
    mapping = []
