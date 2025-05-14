import sys
import GlobalVars as GV
import argparse

import numpy as np

import Init
import copy
import time

import Task
import csv
import os
import pickle
from Inst_Parser import parse_inst_file

from math import log2, ceil

# NOTE
# Compiler do the followings:
# 1) save the simulator config in sim_params.pkl
# 2) save the HW architectural state in arch_state.pkl

# for initializing memory layout
def get_num_entries(core_ind, num_entries, initial_mem, is_hist = False, hist_metadata = None, hist_pos = None):
    for ltask_id in range(GV.NUM_SCHEDULED_TASKS[core_ind]):
        gtask_id = GV.ltask_to_gtask[core_ind][ltask_id]
        temp_dict = {}
        if is_hist:
            # The memory lines per unit
            unit_offset = {}
            # precision of each data
            log2_precision = {}
            # length of hist
            length = {}
        for key in initial_mem[gtask_id].keys():
            if is_hist:
                unit_offset[key] = 0
                log2_precision[key] = ceil(log2(initial_mem[gtask_id][key][1]))
                length[key] = 0
                memory_data = initial_mem[gtask_id][key][0]
                hist_pos.append({'val': -1, 'task' : ltask_id, 'data' : key})
                if len(memory_data) == 0:
                    temp_dict[key] = len(memory_data)
                else:
                    unit_offset[key] = len(memory_data[0])
                    length[key] = len(memory_data[0][0]) * (unit_offset[key] - 1) + len(memory_data[0][-1])
                    # flatten 2d initial_mem into 1d list & overwrite the flattened data to the tuple
                    initial_mem[gtask_id][key] = sum(initial_mem[gtask_id][key][0], [])
                    temp_dict[key] = len(initial_mem[gtask_id][key])
            else:
                temp_dict[key] = len(initial_mem[gtask_id][key])
        if is_hist:
            hist_metadata[ltask_id]['unit_offset'] = unit_offset
            hist_metadata[ltask_id]['log2_precision'] = log2_precision
            hist_metadata[ltask_id]['length'] = length
        num_entries.append(temp_dict)
    if is_hist:
        hist_pos += [{'val': -1, 'task' : -1, 'data' : None} for _ in range(64 - len(hist_pos))]

def init_mem(core_ind, mem_num_entries, mem_offset, initial_mem,
             mem, is_forward = False, is_corr = False):

    current_offset = 0
    current_data_offset = [0]
    offset = 0
    for ltask_id in range(GV.NUM_SCHEDULED_TASKS[core_ind]):
        gtask_id = GV.ltask_to_gtask[core_ind][ltask_id]
        for key in mem_num_entries[ltask_id].keys():
            entry = initial_mem[gtask_id][key]
            mem_offset[ltask_id][key] = current_offset
            # Set the width
            width = mem_num_entries[ltask_id][key]
            width_total = width
            for ind in range(width):
                if is_forward:
                    if is_corr:
                        new_entry = (entry[ind][0] + current_data_offset[-1], entry[ind][1], entry[ind][2])
                    else:
                        new_entry = entry[ind] + current_data_offset[-1]
                    mem.append(new_entry)
                else:
                    mem.append(entry[ind])
            if is_forward:
                if not len(entry) == 0:
                    if is_corr:
                        current_data_offset.append(current_data_offset[-1] + entry[-1][0])
                    else:
                        current_data_offset.append(current_data_offset[-1] + entry[-1])
                else:
                    current_data_offset.append(current_data_offset[-1])

            current_offset += width_total
            offset += 1

    return current_data_offset

np.random.seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('--num_tasks', type=int, dest='num_tasks')
parser.add_argument('--workload_path', nargs='+', type=str, dest='workload_path')
parser.add_argument('--mapping_file_path', nargs='+', type=str, dest='mapping_file_path')
parser.add_argument('--highest_priority', type=int, dest='highest_priority')
parser.add_argument('--task_type', nargs='+', type=str, dest='task_type')
parser.add_argument('--max_core', nargs='+', type=int, dest='max_core')
parser.add_argument('--workload_timestep', nargs='+', type=int, dest='workload_timestep')
parser.add_argument('--dt', nargs='+', type=int, dest='dt')
parser.add_argument('--latency', nargs='+', type=int, dest='latency')

# Fixed for simulator
parser.add_argument('--cyc_period', type=float, dest='cyc_period')
# parser.add_argument('--debug', type=int, dest='debug')
parser.add_argument('--task_to_core', nargs='+', type=str, action='append', dest='task_to_core')
parser.add_argument('--p_unit', nargs='+', type=str, action='append', dest='p_unit')
parser.add_argument('--use_partial', nargs='+', type=int, dest='use_partial')
parser.add_argument('--baseline', type=str, dest='baseline')
parser.add_argument('--optimize', type=str, dest='optimize')

parser.add_argument('--no_cascade', type=str, dest='no_cascade')
parser.add_argument('--external', type=str, dest='external')
parser.add_argument('--working_directory', type=str, dest='working_directory')
parser.add_argument('--simulator_path', type=str, dest='simulator_path')
parser.add_argument('--executable_path', type=str, dest='executable_path')
parser.add_argument('--saved_arch_states', type=str, dest='saved_arch_states')
args = parser.parse_args()

# Set common parameter
GV.sim_params['max_core_x'] = args.max_core[0]
GV.sim_params['max_core_y'] = args.max_core[1]
GV.sim_params['used_core_num'] = GV.sim_params['max_core_x'] * GV.sim_params['max_core_y']

# for debugging
GV.sim_params['cyc_period'] = args.cyc_period

# divide 1ms by the cycle period
GV.sim_params['1ms_cyc'] = int(1e6 // args.cyc_period)

workload_path = args.workload_path
mapping_file_name = args.mapping_file_path
GV.sim_params['highest_priority'] = args.highest_priority

GV.sim_params['workload_timestep'] = args.workload_timestep
GV.sim_params['dt'] = args.dt
GV.sim_params['latency'] = args.latency
GV.sim_params['use_partial'] = args.use_partial

GV.sim_params['violation_ratio'] = {'PASSED' : 0, 'FAILED' : 0}

# The set the external core and its parent core
GV.sim_params['external'] = (args.external == 'True')

# Save the working directory & simulator path
GV.sim_params['working_directory'] = args.working_directory
GV.sim_params['simulator_path'] = args.simulator_path

GV.sim_params['baseline'] = (args.baseline == 'True')
GV.sim_params['optimize'] = (args.optimize == 'True')
GV.sim_params['no_cascade'] = (args.no_cascade == 'True')

# Register file configuration
GV.reg_event_size = 10
GV.reg_free_region = 30
GV.reg_general_region_size = GV.reg_event_size + GV.reg_free_region

# Set the tasks
for task_id in range(args.num_tasks):
    dat = args.task_type[task_id]
    Task.add_task(dat)

###############################################################
# Parse into a nested list
GV.ltask_to_gtask = None
GV.gtask_to_ltask = None
GV.sim_params['task_to_core'] = None
GV.sim_params['core_to_task'] = None
core_list = ""
for item in args.task_to_core[0]:
    core_list += item
initial_task_translation_total = Init.init_task_mem(core_list)

# Parse into a dict
p_unit_list = ""
for item in args.p_unit[0]:
    p_unit_list += item
p_unit = []
for item in eval(p_unit_list):
    p_unit.append(item)
GV.sim_params['p_unit'] = p_unit

print("Start Connection Initialization\n")
sys.stdout.flush()

# Make duplicate according to the task
GV.sim_params['num_unit'] = [0 for _ in range(GV.TOTAL_TASKS)]
GV.sim_params['num_unit_types'] = [0 for _ in range(GV.TOTAL_TASKS)]
GV.sim_params['unit_type_name'] = [0 for _ in range(GV.TOTAL_TASKS)]
GV.sim_params['num_total_unit'] = [0 for _ in range(GV.TOTAL_TASKS)]
GV.sim_params['num_unit_per_core'] = [0 for _ in range(GV.TOTAL_TASKS)]
GV.sim_params['consts'] = [None for _ in range(GV.TOTAL_TASKS)]
GV.loop_info = [{} for _ in range(GV.TOTAL_TASKS)]
GV.lid_to_gid = [None for _ in range(GV.TOTAL_TASKS)]
GV.pid_to_gid = [None for _ in range(GV.TOTAL_TASKS)]
GV.tid_to_core = [None for _ in range(GV.TOTAL_TASKS)]
GV.gid_to_core = [None for _ in range(GV.TOTAL_TASKS)]

GV.leading_timestep = [-1 for _ in range(GV.TOTAL_TASKS)]
GV.per_core_timestep = [[-1 for _ in range(GV.sim_params['used_core_num'])] for _ in range(GV.TOTAL_TASKS)]
initial_hist_mem_total  = [[] for _ in range(GV.sim_params['used_core_num'])]
initial_state_mem_total  = [[] for _ in range(GV.sim_params['used_core_num'])]
#initial_stack_mem_total  = [[] for _ in range(GV.sim_params['used_core_num'])]
initial_ack_left_mem_total  = [[] for _ in range(GV.sim_params['used_core_num'])]
initial_ack_num_mem_total  = [[] for _ in range(GV.sim_params['used_core_num'])]
initial_ack_stack_mem_total  = [[] for _ in range(GV.sim_params['used_core_num'])]

initial_corr_forward_total  = [[] for _ in range(GV.sim_params['used_core_num'])]
initial_corr_mem_total  = [[] for _ in range(GV.sim_params['used_core_num'])]
initial_route_forward_total = [[] for _ in range(GV.sim_params['used_core_num'])]
initial_route_mem_total = [[] for _ in range(GV.sim_params['used_core_num'])]
initial_sync_info_total = [[] for _ in range(GV.sim_params['used_core_num'])]
external_input_total    = []

# header length : core_num + 4
memory_footprint_list_header = ['task', 'memory type'] + [f'core {i}' for i in range(GV.sim_params['used_core_num'])] + ['average', 'total']
memory_footprint_list = [memory_footprint_list_header]
# open file for memory footprint
memory_footprint_file = open('memory_footprint.csv', 'w', newline='')

for gtask_id in range(GV.TOTAL_TASKS):
    Init.init_network(workload_path[gtask_id] + 'network_parameter.npy', gtask_id)

    task_type = Task.get_task(gtask_id).type

    # initialize corr / route memory
    # save corr / route memory footprint in a file
    # initial_corr_forward, initial_corr_inverse_indirect, initial_corr_inverse_mem, initial_corr_mem, \
    initial_corr_forward, initial_corr_mem, \
        initial_route_forward, initial_route_mem \
        = Init.init_corr_mem(
        workload_path[gtask_id] + 'connection.npy',
        mapping_file_name[gtask_id], GV.sim_params['task_to_core'][gtask_id], gtask_id)

    # 1. corr memory footprint
    corr_memory_list = ['corr_forward', 'corr_mem', 'route_forward', 'route_mem']
    initial_corr_list = [initial_corr_forward, initial_corr_mem, initial_route_forward, initial_route_mem]
    for corr_memory_name, initial_corr_memory in zip(corr_memory_list, initial_corr_list):
        corr_mf = [task_type, corr_memory_name]
        total_mf = 0
        for core_id in range(GV.sim_params['used_core_num']):
            # calc memory footprint per core
            mf = 0
            for corr_name in initial_corr_memory[core_id].keys():
                for corr_entry in initial_corr_memory[core_id][corr_name]:
                    mf += GV.MEM_WIDTH[corr_memory_name]
            total_mf += mf
            corr_mf.append(mf)

        avg_mf = total_mf / GV.sim_params['used_core_num']
        corr_mf.append(avg_mf)
        corr_mf.append(total_mf)
        memory_footprint_list.append(corr_mf)

    # initialize other memory
    # save other memory (hist_mem, stack_mem, ack_left_mem, ack_num_mem, ack_stack_mem) footprint in a file
    initial_state_mem, initial_hist_mem, initial_ack_left_mem, \
    initial_ack_num_mem, initial_ack_stack_mem \
        = Init.init_other_mem(workload_path[gtask_id] + 'initial_states.npy',
                              task_type, gtask_id)

    # 2. other memory footprint
    other_memory_list = ['state_mem', 'hist_mem', 'ack_left_mem', 'ack_num_mem', 'ack_stack_mem']
    initial_other_list = [initial_state_mem, initial_hist_mem, initial_ack_left_mem, initial_ack_num_mem, initial_ack_stack_mem]
    for other_memory_name, initial_other_memory in zip(other_memory_list, initial_other_list):
        other_mf = [task_type, other_memory_name]
        total_mf = 0
        for core_id in range(GV.sim_params['used_core_num']):
            # calc memory footprint per core
            mf = 0
            for other_name in initial_other_memory[core_id].keys():
                for other_entry in initial_other_memory[core_id][other_name]:
                    if isinstance(other_entry, list):
                        mf += (GV.MEM_WIDTH[other_memory_name] * len(other_entry))
                    else:
                        mf += GV.MEM_WIDTH[other_memory_name]
            total_mf += mf
            other_mf.append(mf)
        avg_mf = total_mf / GV.sim_params['used_core_num']
        other_mf.append(avg_mf)
        other_mf.append(total_mf)
        memory_footprint_list.append(other_mf)

    external_input = Init.load_external_stimulus(workload_path[gtask_id] + 'external_stimulus.npy')
    initial_sync_info = Init.init_spike_sync_topology(initial_route_forward, initial_route_mem, GV.sim_params['task_to_core'][gtask_id], GV.sim_params['external'])

    # NOTE : set task consts here
    # save task consts
    if task_type == 'snn' or task_type == 'ann':
        pass

    elif task_type == 'ss':
        # read consts from network params / preprocess consts if necessary
        n_t = GV.sim_params['consts'][gtask_id]['n_t']
        n_scalar_neg = -1/GV.sim_params['consts'][gtask_id]['n_scalar']
        template_shift = GV.sim_params['consts'][gtask_id]['template_shift']
        # save task consts to use
        Task.set_const(gtask_id, 'n_t', n_t)
        Task.set_const(gtask_id, 'n_scalar_neg', n_scalar_neg)
        Task.set_const(gtask_id, 'template_shift', template_shift)

    elif task_type == 'tm':
        # read consts from network params / preprocess consts if necessary
        temp_width = GV.sim_params['consts'][gtask_id]['n_t']
        # save task consts to use
        Task.set_const(gtask_id, 'temp_width', temp_width)

    elif task_type == 'pc':
        # read consts from network params / preprocess consts if necessary
        window = GV.sim_params['consts'][gtask_id]['window'] + 1
        corr_period = GV.sim_params['consts'][gtask_id]['corr_period']
        # save task consts to use
        Task.set_const(gtask_id, 'window', window)
        Task.set_const(gtask_id, 'corr_period', corr_period)

    # Append to the total list
    external_input_total.append(external_input)
    for core_id in range(GV.sim_params['used_core_num']):
        initial_state_mem_total[core_id].append(initial_state_mem[core_id])
        initial_hist_mem_total[core_id].append(initial_hist_mem[core_id])
        #initial_stack_mem_total[core_id].append(initial_stack_mem[core_id])
        initial_ack_left_mem_total[core_id].append(initial_ack_left_mem[core_id])
        initial_ack_num_mem_total[core_id].append(initial_ack_num_mem[core_id])
        initial_ack_stack_mem_total[core_id].append(initial_ack_stack_mem[core_id])
        initial_corr_forward_total[core_id].append(initial_corr_forward[core_id])
        initial_corr_mem_total[core_id].append(initial_corr_mem[core_id])
        initial_route_forward_total[core_id].append(initial_route_forward[core_id])
        initial_route_mem_total[core_id].append(initial_route_mem[core_id])
        initial_sync_info_total[core_id].append(initial_sync_info[core_id])

print("Save memory footprint file")
writer = csv.writer(memory_footprint_file)
writer.writerows(memory_footprint_list)
memory_footprint_file.close()
print("End saving memory footprint file")
print("End Connection Initialization\n")
sys.stdout.flush()

simulator_executables_path = args.executable_path + "/simulator_executables"
if not os.path.isdir(simulator_executables_path):
    os.mkdir(simulator_executables_path)

# 1. save simulation params
with open(simulator_executables_path + '/sim_params.pkl', 'wb') as f:
    pickle.dump(GV.sim_params, f)

# save task-related parameters
task_params = {}

task_params['MAX_LOCAL_TASK'] = GV.MAX_LOCAL_TASK
task_params['NUM_SCHEDULED_TASKS'] = GV.NUM_SCHEDULED_TASKS
task_params['NUM_COMPLETED_TASKS'] = GV.NUM_COMPLETED_TASKS
task_params['TOTAL_TASKS'] = GV.TOTAL_TASKS

task_params['ltask_to_gtask'] = GV.ltask_to_gtask
task_params['gtask_to_ltask'] = GV.gtask_to_ltask

task_params['total_info'] = Task.total_info
task_params['tasks'] = Task.tasks

task_params['per_core_timestep'] = GV.per_core_timestep
task_params['leading_timestep'] = GV.leading_timestep

task_params['loop_info'] = GV.loop_info
task_params['lid_to_gid'] = GV.lid_to_gid
task_params['pid_to_gid'] = GV.pid_to_gid
task_params['tid_to_core'] = GV.tid_to_core
task_params['gid_to_core'] = GV.gid_to_core

with open(simulator_executables_path + '/task_params.pkl', 'wb') as f:
    pickle.dump(task_params, f)

ext_params = {}
ext_params['external_id'] = GV.external_id
ext_params['reg_event_size'] = GV.reg_event_size
ext_params['reg_free_region'] = GV.reg_free_region
ext_params['reg_general_region_size'] = GV.reg_general_region_size

with open(simulator_executables_path + '/ext_params.pkl', 'wb') as f:
    pickle.dump(ext_params, f)

# 2. save initial architectural state

# initialize memory layout and save

########## MEMORY LAYOUT INITIALIZE START ###########

state_mem_num_entries = [[] for core_ind in range(GV.sim_params['used_core_num'])]
state_mem_offset = [[{} for ltask_id in range(GV.NUM_SCHEDULED_TASKS[core_ind])] for core_ind in range(GV.sim_params['used_core_num'])]
state_mem = [[] for core_ind in range(GV.sim_params['used_core_num'])]

for core_ind in range(GV.sim_params['used_core_num']):
    get_num_entries(core_ind, state_mem_num_entries[core_ind], initial_state_mem_total[core_ind])
    init_mem(core_ind, state_mem_num_entries[core_ind], state_mem_offset[core_ind], initial_state_mem_total[core_ind], state_mem[core_ind])

# hist mem
hist_mem_num_entries = [[] for core_ind in range(GV.sim_params['used_core_num'])]
hist_metadata = [[{} for ltask_id in range(GV.NUM_SCHEDULED_TASKS[core_ind])] for core_ind in range(GV.sim_params['used_core_num'])]
hist_pos = [[] for core_ind in range(GV.sim_params['used_core_num'])]
hist_mem_offset = [[{} for ltask_id in range(GV.NUM_SCHEDULED_TASKS[core_ind])] for core_ind in range(GV.sim_params['used_core_num'])]
hist_mem = [[] for core_ind in range(GV.sim_params['used_core_num'])]

hist_task_translator = [[{} for _ in range(GV.MAX_LOCAL_TASK)] for core_ind in range(GV.sim_params['used_core_num'])]
for core_ind in range(GV.sim_params['used_core_num']):
    get_num_entries(core_ind, hist_mem_num_entries[core_ind], initial_hist_mem_total[core_ind], True, hist_metadata[core_ind], hist_pos[core_ind])
    init_mem(core_ind, hist_mem_num_entries[core_ind], hist_mem_offset[core_ind], initial_hist_mem_total[core_ind], hist_mem[core_ind])

    index = 0
    for ltask_id in range(len(hist_metadata[core_ind])):
        for key in hist_metadata[core_ind][ltask_id]['unit_offset'].keys():
            hist_task_translator[core_ind][ltask_id][key] = index
            index += 1

# stack mem
#stack_mem_num_entries = [[] for core_ind in range(GV.sim_params['used_core_num'])]
#stack_mem_offset = [[{} for ltask_id in range(GV.NUM_SCHEDULED_TASKS[core_ind])] for core_ind in range(GV.sim_params['used_core_num'])]
#stack_mem = [[] for core_ind in range(GV.sim_params['used_core_num'])]
#
#for core_ind in range(GV.sim_params['used_core_num']):
#    get_num_entries(core_ind, stack_mem_num_entries[core_ind], initial_stack_mem_total[core_ind])
#    init_mem(core_ind, stack_mem_num_entries[core_ind], stack_mem_offset[core_ind], initial_stack_mem_total[core_ind], stack_mem[core_ind])
#
#stack_ptr = [[{key : 0 for key in initial_stack_mem_total[core_ind][GV.ltask_to_gtask[core_ind][ltask_id]].keys()} for ltask_id in range(GV.NUM_SCHEDULED_TASKS[core_ind])] for core_ind in range(GV.sim_params['used_core_num'])]

# corr mem
corr_mem_num_entries = [[] for core_ind in range(GV.sim_params['used_core_num'])]
corr_mem_offset = [[{} for ltask_id in range(GV.NUM_SCHEDULED_TASKS[core_ind])] for core_ind in range(GV.sim_params['used_core_num'])]
corr_mem = [[] for core_ind in range(GV.sim_params['used_core_num'])]

for core_ind in range(GV.sim_params['used_core_num']):
    get_num_entries(core_ind, corr_mem_num_entries[core_ind], initial_corr_mem_total[core_ind])
    init_mem(core_ind, corr_mem_num_entries[core_ind], corr_mem_offset[core_ind], initial_corr_mem_total[core_ind], corr_mem[core_ind])

# corr_forward mem
corr_forward_num_entries = [[] for core_ind in range(GV.sim_params['used_core_num'])]
corr_forward_offset = [[{} for ltask_id in range(GV.NUM_SCHEDULED_TASKS[core_ind])] for core_ind in range(GV.sim_params['used_core_num'])]
corr_forward = [[] for core_ind in range(GV.sim_params['used_core_num'])]

for core_ind in range(GV.sim_params['used_core_num']):
    get_num_entries(core_ind, corr_forward_num_entries[core_ind], initial_corr_forward_total[core_ind])
    init_mem(core_ind, corr_forward_num_entries[core_ind], corr_forward_offset[core_ind], initial_corr_forward_total[core_ind], corr_forward[core_ind], is_forward=True, is_corr=True)

# route forward mem
route_forward_num_entries = [[] for core_ind in range(GV.sim_params['used_core_num'])]
route_forward_offset = [[{} for ltask_id in range(GV.NUM_SCHEDULED_TASKS[core_ind])] for core_ind in range(GV.sim_params['used_core_num'])]
route_forward = [[] for core_ind in range(GV.sim_params['used_core_num'])]

for core_ind in range(GV.sim_params['used_core_num']):
    get_num_entries(core_ind, route_forward_num_entries[core_ind], initial_route_forward_total[core_ind])
    init_mem(core_ind, route_forward_num_entries[core_ind], route_forward_offset[core_ind], initial_route_forward_total[core_ind], route_forward[core_ind], is_forward=True)

# route mem
route_mem_num_entries = [[] for core_ind in range(GV.sim_params['used_core_num'])]
route_mem_offset = [[{} for ltask_id in range(GV.NUM_SCHEDULED_TASKS[core_ind])] for core_ind in range(GV.sim_params['used_core_num'])]
route_mem = [[] for core_ind in range(GV.sim_params['used_core_num'])]

for core_ind in range(GV.sim_params['used_core_num']):
    get_num_entries(core_ind, route_mem_num_entries[core_ind], initial_route_mem_total[core_ind])
    init_mem(core_ind, route_mem_num_entries[core_ind], route_mem_offset[core_ind], initial_route_mem_total[core_ind], route_mem[core_ind])

ack_left_mem_num_entries = [[] for core_ind in range(GV.sim_params['used_core_num'])]
ack_left_mem_offset = [[{} for ltask_id in range(GV.NUM_SCHEDULED_TASKS[core_ind])] for core_ind in range(GV.sim_params['used_core_num'])]
ack_left_mem = [[] for core_ind in range(GV.sim_params['used_core_num'])]

for core_ind in range(GV.sim_params['used_core_num']):
    get_num_entries(core_ind, ack_left_mem_num_entries[core_ind], initial_ack_left_mem_total[core_ind])
    init_mem(core_ind, ack_left_mem_num_entries[core_ind], ack_left_mem_offset[core_ind], initial_ack_left_mem_total[core_ind], ack_left_mem[core_ind])

ack_num_mem_num_entries = [[] for core_ind in range(GV.sim_params['used_core_num'])]
ack_num_mem_offset = [[{} for ltask_id in range(GV.NUM_SCHEDULED_TASKS[core_ind])] for core_ind in range(GV.sim_params['used_core_num'])]
ack_num_mem = [[] for core_ind in range(GV.sim_params['used_core_num'])]

for core_ind in range(GV.sim_params['used_core_num']):
    get_num_entries(core_ind, ack_num_mem_num_entries[core_ind], initial_ack_num_mem_total[core_ind])
    init_mem(core_ind, ack_num_mem_num_entries[core_ind], ack_num_mem_offset[core_ind], initial_ack_num_mem_total[core_ind], ack_num_mem[core_ind])

ack_stack_mem_num_entries = [[] for core_ind in range(GV.sim_params['used_core_num'])]
ack_stack_mem_offset = [[{} for ltask_id in range(GV.NUM_SCHEDULED_TASKS[core_ind])] for core_ind in range(GV.sim_params['used_core_num'])]
ack_stack_mem = [[] for core_ind in range(GV.sim_params['used_core_num'])]

for core_ind in range(GV.sim_params['used_core_num']):
    get_num_entries(core_ind, ack_stack_mem_num_entries[core_ind], initial_ack_stack_mem_total[core_ind])
    init_mem(core_ind, ack_stack_mem_num_entries[core_ind], ack_stack_mem_offset[core_ind], initial_ack_stack_mem_total[core_ind], ack_stack_mem[core_ind])

ack_stack_ptr = [[{key : 0 for key in initial_ack_stack_mem_total[core_ind][GV.ltask_to_gtask[core_ind][ltask_id]].keys()} for ltask_id in range(GV.NUM_SCHEDULED_TASKS[core_ind])] for core_ind in range(GV.sim_params['used_core_num'])]

GV.state_mem_offset = state_mem_offset

# save mem_offset in GlobalVars (to use in Inst_Parser)
GV.hist_mem_offset = hist_mem_offset
#GV.stack_mem_offset = stack_mem_offset
GV.corr_forward_offset = corr_forward_offset


saved_arch_states = args.saved_arch_states.split(',')
arch_state = {}

arch_state['initial_timestep'] = []
if len(saved_arch_states) < GV.TOTAL_TASKS:
    print("Not using saved architectural states")
    saved_arch_states = ['None' for _ in range(GV.TOTAL_TASKS)]

for gtask_id in range(GV.TOTAL_TASKS):
    if saved_arch_states[gtask_id] != 'None':
        # load the saved arch state if exists
        print("Using saved architectural states for task", gtask_id, ': saved_arch_states/' + saved_arch_states[gtask_id] + '.pkl\n')
        with open(simulator_executables_path + '/saved_arch_states/' + saved_arch_states[gtask_id] + '.pkl', 'rb') as f:
            saved_arch_state = pickle.load(f)

        for core_ind in GV.sim_params['task_to_core'][gtask_id]:
                ltask_id = GV.gtask_to_ltask[gtask_id][core_ind]
                state_mem[core_ind][list(state_mem_offset[core_ind][ltask_id].values())[0] : list(state_mem_offset[core_ind][ltask_id].values())[0] + len(saved_arch_state['state_mem'][core_ind])] \
                = saved_arch_state['state_mem'][core_ind]
                hist_mem[core_ind][list(hist_mem_offset[core_ind][ltask_id].values())[0] : list(hist_mem_offset[core_ind][ltask_id].values())[0] + len(saved_arch_state['hist_mem'][core_ind])] \
                = saved_arch_state['hist_mem'][core_ind]
                corr_mem[core_ind][list(corr_mem_offset[core_ind][ltask_id].values())[0] : list(corr_mem_offset[core_ind][ltask_id].values())[0] + len(saved_arch_state['corr_mem'][core_ind])] \
                = saved_arch_state['corr_mem'][core_ind]

                for pos in saved_arch_state['hist_pos'][core_ind]:
                    if pos['task'] == -1:
                        break
                    assert(pos['task'] == 0)
                    hist_pos[core_ind][hist_task_translator[core_ind][ltask_id][pos['data']]] = {'val': pos['val'], 'task': ltask_id, 'data': pos['data']}

        arch_state['initial_timestep'].append(saved_arch_state['initial_timestep'][0])
    else:
        arch_state['initial_timestep'].append(0)

########## MEMORY LAYOUT INITIALIZE END #############
arch_state['state_mem'] = state_mem
arch_state['state_mem_offset'] = state_mem_offset

arch_state['hist_mem_num_entries'] = hist_mem_num_entries
arch_state['hist_metadata'] = hist_metadata
arch_state['hist_pos'] = hist_pos
arch_state['hist_mem_offset'] = hist_mem_offset
arch_state['hist_mem'] = hist_mem

## repeat for other memory types
#arch_state['stack_mem_num_entries'] = stack_mem_num_entries
#arch_state['stack_mem_offset'] = stack_mem_offset
#arch_state['stack_mem'] = stack_mem
#arch_state['stack_ptr'] = stack_ptr

arch_state['corr_mem_num_entries'] = corr_mem_num_entries
arch_state['corr_mem_offset'] = corr_mem_offset
arch_state['corr_mem'] = corr_mem

arch_state['corr_forward_num_entries'] = corr_forward_num_entries
arch_state['corr_forward_offset'] = corr_forward_offset
arch_state['corr_forward'] = corr_forward

arch_state['route_forward_num_entries'] = route_forward_num_entries
arch_state['route_forward_offset'] = route_forward_offset
arch_state['route_forward'] = route_forward

arch_state['route_mem_num_entries'] = route_mem_num_entries
arch_state['route_mem_offset'] = route_mem_offset
arch_state['route_mem'] = route_mem

arch_state['ack_left_mem_num_entries'] = ack_left_mem_num_entries
arch_state['ack_left_mem_offset'] = ack_left_mem_offset
arch_state['ack_left_mem'] = ack_left_mem

arch_state['ack_num_mem_num_entries'] = ack_num_mem_num_entries
arch_state['ack_num_mem_offset'] = ack_num_mem_offset
arch_state['ack_num_mem'] = ack_num_mem

arch_state['ack_stack_mem_num_entries'] = ack_stack_mem_num_entries
arch_state['ack_stack_mem_offset'] = ack_stack_mem_offset
arch_state['ack_stack_mem'] = ack_stack_mem
arch_state['ack_stack_ptr'] = ack_stack_ptr

arch_state['initial_sync_info_total'] = initial_sync_info_total
arch_state['initial_task_translation_total'] = initial_task_translation_total
arch_state['external_input_total'] = external_input_total

# build PC controller's initial architectural state
table_key_list_total = [[] for _ in range(GV.sim_params['used_core_num'])]
base_list_total = [[] for _ in range(GV.sim_params['used_core_num'])]
bound_list_total = [[] for _ in range(GV.sim_params['used_core_num'])]
inst_list_total = [[] for _ in range(GV.sim_params['used_core_num'])]

# Check valid condition
if GV.sim_params['baseline']:
    assert(GV.sim_params['optimize'] == False)

for core_ind in range(GV.sim_params['used_core_num']):
    for ltask_id in range(GV.NUM_SCHEDULED_TASKS[core_ind]):
        gtask_id = GV.ltask_to_gtask[core_ind][ltask_id]
        #parsed_graph_inst = parse_inst_file('instruction_api/tm_bci_send.inst', gtask_id, core_ind)
        #assert(0)
        for event_type in Task.get_task(gtask_id).event:
            if GV.sim_params['baseline']:
                parsed_graph_inst = parse_inst_file('instruction_isa/{}.inst'.format(event_type), gtask_id, core_ind)
            elif GV.sim_params['optimize']:
                parsed_graph_inst = parse_inst_file('instruction_pipelined/{}.inst'.format(event_type), gtask_id, core_ind)
            else:
                parsed_graph_inst = parse_inst_file('instruction_loopctrl/{}.inst'.format(event_type), gtask_id, core_ind)
            #parsed_graph_inst = parse_inst_file('instruction_api_loopctrl/{}.inst'.format(event_type), gtask_id, core_ind)
            table_key_list_total[core_ind].append((ltask_id, event_type))
            base = len(inst_list_total[core_ind])
            size = len(parsed_graph_inst)
            base_list_total[core_ind].append(base)
            bound_list_total[core_ind].append(base + size)
            for idx in range(size):
                inst_list_total[core_ind].append(parsed_graph_inst[idx])

arch_state['table_key_list_total'] = table_key_list_total
arch_state['base_list_total'] = base_list_total
arch_state['bound_list_total'] = bound_list_total
arch_state['inst_list_total'] = inst_list_total

with open(simulator_executables_path + '/arch_state.pkl', 'wb') as f:
    pickle.dump(arch_state, f)
