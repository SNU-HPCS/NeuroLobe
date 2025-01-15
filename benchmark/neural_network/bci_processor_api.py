import os 
import sys

import numpy as np
import math

import networkx as nx 
from elephant.spike_train_generation import homogeneous_poisson_process

#
network_params = {}
neu_states = None


simtime_dat = None
dt_dat = None

conn_list = []
offset_list = []
mapping_list = []
external_stimulus = None

def set_seed(seed: int):
    np.random.seed(seed)

def init_simulation(simtime: float,
                    dt: float,
                    num_neurons: int,
                    num_external: int,
                    result_path):
    global simtime_dat
    global dt_dat
    global network_params
    global offset_list
    global result_dir
    
    result_dir = result_path

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    simtime_dat = simtime
    dt_dat = dt
    network_params["simtime"] = int(simtime / dt)

    network_params["num_internal"] = num_neurons + num_external
    network_params["num_unit"] = [num_neurons, num_external, num_external]
    network_params["num_unit_types"] = 3
    network_params["unit_types"] = ["neuron", "bci_neuron", "bci"]
    offset_list = [0, num_neurons, num_neurons + num_external]

    print("Initialized Simulation")
    sys.stdout.flush()

def create_external_stimulus(num_external: int,
                             external_stim: list):

    global network_params
    global external_stimulus

    network_params["num_external"] = num_external
    external_stimulus = external_stim

    print("Created External Stimulus")
    sys.stdout.flush()


def create_neurons(initial_states: list):

    global neu_states

    neu_states = initial_states

    print("Created Neurons")
    sys.stdout.flush()


def create_connections(conn_dat: list, num_types:int):

    global conn_list
    global mapping_list
    global offset_list

    conn_list = [[[] for _ in range(num_types)] for _ in range(num_types)]

    for t1 in range(num_types):
        for t2 in range(num_types):
            src_offset = offset_list[t1]
            dst_offset = offset_list[t2]
            data_list = conn_dat[t1][t2]
            if data_list:
                for data in data_list:
                    src = data[0]
                    dst = data[1]
                    if t1 == 2:
                        conn_list[t1][t2].append((src, dst, False, None))
                    else:
                        conn_list[t1][t2].append((src, dst, True, data[2]))
                    if t1 < num_types - 1 and t2 < num_types - 1:
                        mapping_list.append((src + src_offset, dst + dst_offset))

    print("Created Connections")
    sys.stdout.flush()


def end_simulation():

    global simtime_dat
    global dt_dat
    
    global conn_list
    global mapping_list
    global network_params
    global neu_states
    #global external_stimulus
    global conn_list
    global external_stimulus

    # check if the network has been properly initialized
    assert(neu_states)
    #assert(external_stimulus)
    assert(len(conn_list))
    
    # set bit precision for hist mem
    neu_states['init_prec'] = {'spike weight accum': 16, 'ext weight accum': 16, 'weight accum': 16, 'bci neuron history': 4, 'bci': 10}

    np.save(result_dir + "/external_stimulus.npy", external_stimulus)
    np.save(result_dir + "/network_parameter.npy", network_params)
    np.save(result_dir + "/initial_states.npy", neu_states)
    conn_list = np.array(conn_list, dtype=object)
    np.save(result_dir + "/connection.npy", conn_list)
    np.save(result_dir + "/mapping.npy", mapping_list)

    print("Generated Required Metadata")
    sys.stdout.flush()

    network_params = {}
    neu_states = None
    external_stimulus = None
    
    
    simtime_dat = None
    dt_dat = None
    
    conn_list = []
