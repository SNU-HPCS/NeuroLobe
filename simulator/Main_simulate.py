import sys
import Core
import NoC
import GlobalVars as GV
import argparse
import os

import numpy as np

# import Init
import copy
import time

import Task
import Profiler
from cProfile import Profile
from pstats import Stats

import csv

from Energy import calculate_energy, save_op_count

import pickle

def init():
    np.random.seed(1)

    # load simulation params
    with open(simulator_executables_path + '/sim_params.pkl', 'rb') as f:
        loaded_sim_params = pickle.load(f)

    GV.sim_params.update(loaded_sim_params)

    with open(simulator_executables_path + '/task_params.pkl', 'rb') as f:
        loaded_task_params = pickle.load(f)

    GV.MAX_LOCAL_TASK = loaded_task_params['MAX_LOCAL_TASK']
    GV.NUM_SCHEDULED_TASKS = loaded_task_params['NUM_SCHEDULED_TASKS']
    GV.NUM_COMPLETED_TASKS = loaded_task_params['NUM_COMPLETED_TASKS']
    GV.TOTAL_TASKS = loaded_task_params['TOTAL_TASKS']

    GV.ltask_to_gtask = loaded_task_params['ltask_to_gtask']
    GV.gtask_to_ltask = loaded_task_params['gtask_to_ltask']

    Task.total_info = loaded_task_params['total_info']
    Task.tasks = loaded_task_params['tasks']

    GV.per_core_timestep = loaded_task_params['per_core_timestep']
    GV.leading_timestep = loaded_task_params['leading_timestep']

    GV.loop_info = loaded_task_params['loop_info']
    GV.lid_to_gid = loaded_task_params['lid_to_gid']
    GV.pid_to_gid = loaded_task_params['pid_to_gid']
    GV.tid_to_core = loaded_task_params['tid_to_core']
    GV.gid_to_core = loaded_task_params['gid_to_core']

    with open(simulator_executables_path + '/ext_params.pkl', 'rb') as f:
        loaded_ext_params = pickle.load(f)

    GV.external_id = loaded_ext_params['external_id']
    GV.reg_event_size = loaded_ext_params['reg_event_size']
    GV.reg_free_region = loaded_ext_params['reg_free_region']
    GV.reg_general_region_size = loaded_ext_params['reg_general_region_size']

    # load architectural state (e.g. initial_hist_mem_total, ...)
    with open(simulator_executables_path + '/arch_state.pkl', 'rb') as f:
        loaded_arch_state = pickle.load(f)

    state_mem = loaded_arch_state['state_mem']
    state_mem_offset = loaded_arch_state['state_mem_offset']

    hist_mem_num_entries = loaded_arch_state['hist_mem_num_entries']
    hist_metadata = loaded_arch_state['hist_metadata']
    hist_pos = loaded_arch_state['hist_pos']
    hist_mem_offset = loaded_arch_state['hist_mem_offset']
    hist_mem = loaded_arch_state['hist_mem']

    #stack_mem_num_entries = loaded_arch_state['stack_mem_num_entries']
    #stack_mem_offset = loaded_arch_state['stack_mem_offset']
    #stack_mem = loaded_arch_state['stack_mem']
    #stack_ptr = loaded_arch_state['stack_ptr']

    corr_mem_num_entries = loaded_arch_state['corr_mem_num_entries']
    corr_mem_offset = loaded_arch_state['corr_mem_offset']
    corr_mem = loaded_arch_state['corr_mem']

    corr_forward_num_entries = loaded_arch_state['corr_forward_num_entries']
    corr_forward_offset = loaded_arch_state['corr_forward_offset']
    corr_forward = loaded_arch_state['corr_forward']

    route_mem_num_entries = loaded_arch_state['route_mem_num_entries']
    route_mem_offset = loaded_arch_state['route_mem_offset']
    route_mem = loaded_arch_state['route_mem']

    route_forward_num_entries = loaded_arch_state['route_forward_num_entries']
    route_forward_offset = loaded_arch_state['route_forward_offset']
    route_forward = loaded_arch_state['route_forward']

    ack_left_mem_num_entries = loaded_arch_state['ack_left_mem_num_entries']
    ack_left_mem_offset = loaded_arch_state['ack_left_mem_offset']
    ack_left_mem = loaded_arch_state['ack_left_mem']

    ack_num_mem_num_entries = loaded_arch_state['ack_num_mem_num_entries']
    ack_num_mem_offset = loaded_arch_state['ack_num_mem_offset']
    ack_num_mem = loaded_arch_state['ack_num_mem']

    ack_stack_mem_num_entries = loaded_arch_state['ack_stack_mem_num_entries']
    ack_stack_mem_offset = loaded_arch_state['ack_stack_mem_offset']
    ack_stack_mem = loaded_arch_state['ack_stack_mem']
    ack_stack_ptr = loaded_arch_state['ack_stack_ptr']

    initial_sync_info_total = loaded_arch_state['initial_sync_info_total']
    initial_task_translation_total = loaded_arch_state['initial_task_translation_total']
    external_input_total = loaded_arch_state['external_input_total']
    table_key_list_total = loaded_arch_state['table_key_list_total']
    base_list_total = loaded_arch_state['base_list_total']
    bound_list_total = loaded_arch_state['bound_list_total']
    inst_list_total = loaded_arch_state['inst_list_total']

    # if 'initial_timestep' in loaded_arch_state:
    GV.initial_timestep = loaded_arch_state['initial_timestep']
    if GV.SAVE_ARCH:
        GV.mem_state = [{} for _ in range(GV.sim_params['used_core_num'])]
    

    # initialize HW
    if GV.sim_params['external']:
        GV.NoC = NoC.NoC()
    GV.cyc = 0
    GV.prev_cyc = 0
    GV.cores = []
    for ind in range(GV.sim_params['used_core_num']):
        core = Core.Core(ind,
                         state_mem[ind],
                         state_mem_offset[ind],
                         hist_mem_num_entries[ind],
                         hist_metadata[ind],
                         hist_pos[ind],
                         hist_mem_offset[ind],
                         hist_mem[ind],
                         #stack_mem_num_entries[ind],
                         #stack_mem_offset[ind],
                         #stack_mem[ind],
                         #stack_ptr[ind],
                         corr_mem_num_entries[ind],
                         corr_mem_offset[ind],
                         corr_mem[ind],
                         corr_forward_num_entries[ind],
                         corr_forward_offset[ind],
                         corr_forward[ind],
                         route_forward_num_entries[ind],
                         route_forward_offset[ind],
                         route_forward[ind],
                         route_mem_num_entries[ind],
                         route_mem_offset[ind],
                         route_mem[ind],
                         ack_left_mem_num_entries[ind],
                         ack_left_mem_offset[ind],
                         ack_left_mem[ind],
                         ack_num_mem_num_entries[ind],
                         ack_num_mem_offset[ind],
                         ack_num_mem[ind],
                         ack_stack_mem_num_entries[ind],
                         ack_stack_mem_offset[ind],
                         ack_stack_mem[ind],
                         ack_stack_ptr[ind],
                         initial_sync_info_total[ind],
                         initial_task_translation_total[ind],
                         external_input_total,
                         table_key_list_total[ind],
                         base_list_total[ind],
                         bound_list_total[ind],
                         inst_list_total[ind],
                         GV.sim_params['external'],
                         GV.sim_params['external'] and ind in range(GV.sim_params['used_core_num'] - GV.sim_params['max_core_x'], GV.sim_params['used_core_num']))
        GV.cores.append(core)

    # for debugging
    GV.spike_out = [[[] for _ in range(GV.sim_params['num_total_unit'][gtask_id])] for gtask_id in range(GV.TOTAL_TASKS)]
    state_file = [open('state{}.dat'.format(gtask_id), 'w') for gtask_id in range(GV.TOTAL_TASKS)]
    spike_file = [open('multicore_spike_out{}.dat'.format(gtask_id), 'w') for gtask_id in range(GV.TOTAL_TASKS)]
    result_file = 'profile_result.pkl'
    GV.debug_list = {'spike' : spike_file, 'state' : state_file, 'result' : result_file}

    # for profiling
    Profiler.initialize()

def simulate():
    simulation_end = False
    counter = 0
    GV.prev_time = time.process_time()
    GV.workload_cyc = [0 for _ in range(GV.TOTAL_TASKS)]
    GV.target_cyc = [int(GV.sim_params['latency'][gtask_id] / GV.sim_params['cyc_period']) for \
                     gtask_id in range(GV.TOTAL_TASKS)]
    for core in GV.cores:
        # We should start by appending bci_send for each task
        for gtask_id in range(GV.TOTAL_TASKS):
            if core.ind in GV.sim_params['task_to_core'][gtask_id]:
                ltask_id = GV.gtask_to_ltask[gtask_id][core.ind]
                core.isa_ctrl.init_bci_send(ltask_id, GV.target_cyc[gtask_id])

    # Simulate each cycle one by one
    while not simulation_end:
        # If there are no valid comps => skip the cycle
        GV.valid_advance = False

        # Advance the cores
        for core in GV.cores:
            core.core_advance()
        # Simulate the NoC
        if GV.sim_params['external']:
            GV.NoC.noc_advance()

        # ts_profile => indicates
        # 1) whether a timestep finished ('done')
        # 2) which task finished ('task_id')
        if GV.ts_profile['done']:
            GV.ts_profile['done'] = False

            gtask_id = GV.ts_profile['gtask_id']
            timestep = GV.leading_timestep[gtask_id]

            Profiler.append_timestep(gtask_id)
            cur_time = time.process_time()
            #if GV.sim_params['external']:
            print('task: %d | %d / %d ... ts latency %d (ns), latency budget %.3f (ms), sim time %.3f (ms), elapsed %.3f (s)' \
               % (gtask_id,
                  timestep,
                  GV.sim_params['workload_timestep'][gtask_id],
                  (GV.cyc - max(GV.workload_cyc[gtask_id], GV.prev_cyc)) * GV.sim_params['cyc_period'],
                  (GV.target_cyc[gtask_id] - GV.cyc) * GV.sim_params['cyc_period'] / 1e6,
                  (GV.cyc * GV.sim_params['cyc_period'] / 1e6),
                  cur_time - GV.prev_time))
            #else:
            #    print('task: %d | %d / %d ... sim time %d (ns), tot time %d (ms), elapsed %f (s)' \
            #    % (gtask_id,
            #        timestep,
            #        GV.sim_params['workload_timestep'][gtask_id],
            #        (GV.cyc - max(GV.workload_cyc[gtask_id], GV.prev_cyc) + int(GV.sim_params['dt'][gtask_id] / GV.sim_params['cyc_period'])) * GV.sim_params['cyc_period'],
            #        int(GV.cyc * GV.sim_params['cyc_period'] // 1e6),
            #        cur_time - GV.prev_time))

            if GV.SAVE_ARCH and GV.TOTAL_TASKS == 1 and timestep == GV.sim_params['workload_timestep'][gtask_id]:
                save_arch()
                print("Saved architectural states in runspace\n")     

            ##### Violation Check HERE
            target_cyc = GV.target_cyc[gtask_id]
            if target_cyc > GV.cyc:
                GV.sim_params['violation_ratio']['PASSED'] += 1
            else:
                exc_ns = (GV.cyc - target_cyc) * GV.sim_params['cyc_period']
                print('LATENCY BUDGET FAILED %d (ns)' % exc_ns)
                GV.sim_params['violation_ratio']['FAILED'] += 1

            # Profile total_cyc and violation_cyc
            Profiler.calc_total_cyc(gtask_id, GV.cyc - max(GV.workload_cyc[gtask_id], GV.prev_cyc), timestep)
            Profiler.calc_violation_cyc(gtask_id, GV.cyc - timestep * int(GV.sim_params['dt'][gtask_id] / GV.sim_params['cyc_period']), timestep)

            GV.prev_cyc = GV.cyc
            GV.prev_time = cur_time

        simulation_end = True
        for core in GV.cores:
            if GV.NUM_SCHEDULED_TASKS[core.ind] > len(GV.NUM_COMPLETED_TASKS[core.ind]):
                simulation_end = False
        if simulation_end: break

        # Warp the simulation (if there are no valid computations)
        if not GV.valid_advance:
        #if False:
            # retrieve the minimum pending cycle and execute
            # print(GV.external_modules[0].pending_cyc)
            min_cyc = min(GV.external_modules[0].pending_cyc)
            for ext_core in GV.external_modules:
                min_cyc = min(min_cyc, min(ext_core.pending_cyc))
            GV.cyc = max(min_cyc, GV.cyc + 1)
        else:
            GV.cyc += 1
def stat():
    for gtask_id in range(GV.TOTAL_TASKS):
        if GV.debug_list['spike'][gtask_id]:
            total_units = GV.sim_params['num_total_unit'][gtask_id]
            for neu in range(total_units):
                GV.debug_list['spike'][gtask_id].write(str(neu) + ": " + str(GV.spike_out[gtask_id][neu]) + "\n")

    for gtask_id in range(GV.TOTAL_TASKS):
        total_units = GV.sim_params['num_total_unit'][gtask_id]
        total_spikes = 0.
        for neu in range(total_units):
            total_spikes += len(GV.spike_out[gtask_id][neu])
        print(total_spikes, total_units, GV.sim_params['workload_timestep'][gtask_id])
        print("Rate: {}".format(total_spikes / total_units / GV.sim_params['workload_timestep'][gtask_id]))

    # profile cyc
    if GV.debug_list['result']:
        Profiler.finalize()
        Profiler.add_data(GV.sim_params['dt'], GV.sim_params['latency'], GV.sim_params['cyc_period'], \
                          GV.TOTAL_TASKS, [task.type for task in Task.tasks], [task.event for task in Task.tasks])
        Profiler.profile_result_pandas.to_pickle(GV.debug_list['result'])

    # profile energy
    energy = calculate_energy()
    print("Energy consumption: {} J".format(energy))

    print("Maximum NoC Buffer Size: {}".format(GV.max_buffer_size))

    # profile op_count
    save_op_count()
    print("Saved op_count as op_count.dat")

    print("max buffer size:", GV.max_buffer_size)

def save_arch():
    saved_executables_path = simulator_executables_path + '/saved_arch_states'
    if not os.path.isdir(saved_executables_path):
        os.mkdir(saved_executables_path)

    arch_state = {}

    for core_ind in range(GV.sim_params['used_core_num']):
        mem_state = {}
        GV.cores[core_ind].mem.save_mem(mem_state)

        for key in mem_state.keys():
            if key not in arch_state:
                arch_state[key] = []
            arch_state[key].append(mem_state[key])

    arch_state['initial_timestep'] = [ts + 1 for ts in GV.leading_timestep]

    with open(saved_executables_path + '/' +  GV.sim_params['working_directory'].split('/')[-2] + '.pkl', 'wb') as f:
        pickle.dump(arch_state, f)


def run():
    init()

    print("Initialized Simulation States\n")
    sys.stdout.flush()

    simulate()

    print("Simulation Done\n")
    sys.stdout.flush()

    stat()


parser = argparse.ArgumentParser()
parser.add_argument('--executable_path', type=str, dest='executable_path')
args = parser.parse_args()

simulator_executables_path = args.executable_path + '/simulator_executables'

run()
# profiler = Profile()
# profiler.runcall(run)

# stats = Stats(profiler)
# stats.strip_dirs()
# stats.sort_stats('tottime')
# stats.print_stats()
