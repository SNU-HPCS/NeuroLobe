cimport cython
import numpy as np
cimport numpy as np
from copy import deepcopy
import Task

import GlobalVars as GV
import copy
from libc.math cimport exp, tanh
import sys


cdef class DebugModule:
    def __init__(self, Core.Core core):
        self.core = core
        self.ind = core.ind

        if GV.DEBUG:
            self.pc_corr = [None for _ in range(GV.NUM_SCHEDULED_TASKS[self.ind])]
            for ltask_id in range(GV.NUM_SCHEDULED_TASKS[self.ind]):
                gtask_id = GV.ltask_to_gtask[self.ind][ltask_id]
                if 'pc' in Task.get_task(gtask_id).type:
                    window = Task.get_task(gtask_id).task_const['window']
                    num_total_neuron = GV.sim_params['num_unit'][gtask_id][0]
                    self.pc_corr[gtask_id] = [[[{'valid' : False, 'value' : None} for _ in range(window)] \
                                                  for _ in range(num_total_neuron)] \
                                                  for _ in range(num_total_neuron)]

    cpdef save_tm_corr(self, int tm, x, int ind, index, str_debug, int ltask_id):
        template_type_id = 1
        gtask_id = GV.ltask_to_gtask[self.ind][ltask_id]
        temp_gid = GV.lid_to_gid[gtask_id][self.ind][template_type_id][tm]
        if -1 <= x <= 1:
            GV.spike_out[gtask_id][temp_gid].append([x, ind]) # append PCC value

    cpdef save_snn_spike(self, int spiked, int timestep, int lid, index, str_debug, int ltask_id):
        neuron_type_id = 0
        gtask_id = GV.ltask_to_gtask[self.ind][ltask_id]
        gid = GV.lid_to_gid[gtask_id][self.ind][neuron_type_id][lid]
        if spiked != 0: GV.spike_out[gtask_id][gid].append(timestep)

    cpdef save_snn_state(self, float state, int timestep, int lid, index, str_debug, int ltask_id):
        neuron_type_id = 0
        gtask_id = GV.ltask_to_gtask[self.ind][ltask_id]
        gid = GV.lid_to_gid[gtask_id][self.ind][neuron_type_id][lid]

        # if gid in GV.debug_dst:
        GV.debug_list['state'][gtask_id].write("{}ts: {}, neu: {}, v_t: {:.4f}\n".format(str_debug, timestep, gid, state))

    cpdef save_pc_corr(self, int target_lid, int reference_pid, x, index, str_debug, int ltask_id):
        # This is for debugging
        # Save the correlation result to the external memory
        if not GV.DEBUG: return
        neuron_type_id = 0
        partial_type_id = 1
        gtask_id = GV.ltask_to_gtask[self.ind][ltask_id]
        target_gid = GV.lid_to_gid[gtask_id][self.ind][neuron_type_id][target_lid]

        # If there is a reference_gid
        window = Task.get_task(gtask_id).task_const['window']
        if str_debug == 'neuromorphic':
            # How to set the 
            reference_gid = GV.pid_to_gid[gtask_id][self.ind][partial_type_id][neuron_type_id][reference_pid]
            reference_gid =  GV.sim_params['consts'][gtask_id]['conn_to_src'][reference_gid]
            #reference_gid = reference_pid
        else:
            # neuron to neuron
            reference_gid = GV.pid_to_gid[gtask_id][self.ind][neuron_type_id][neuron_type_id][reference_pid]

        # Convert gid-to-gid

        timestep = GV.per_core_timestep[gtask_id][self.ind]

        idx = timestep // GV.sim_params['consts'][gtask_id]['corr_period']
       
        ## Set dst to src indexing
        self.pc_corr[gtask_id][target_gid][reference_gid][index]['valid'] = True
        self.pc_corr[gtask_id][target_gid][reference_gid][index]['value'] = x

        # Check if all valid
        valid = True
        value_sum = 0
        for i in range(window):
            if self.pc_corr[gtask_id][target_gid][reference_gid][i]['valid']:
                value_sum += self.pc_corr[gtask_id][target_gid][reference_gid][i]['value']
            else:
                valid = False
                break

        if valid:
            while len(GV.spike_out[gtask_id][target_gid]) <= idx:
                GV.spike_out[gtask_id][target_gid].append({})

            if value_sum:
                GV.spike_out[gtask_id][target_gid][idx][reference_gid] = \
                    [self.pc_corr[gtask_id][target_gid][reference_gid][i]['value'] for i in range(window)]
                dict_temp = {}
                tuple_temp = sorted(GV.spike_out[gtask_id][target_gid][idx].items())
                for key, value in tuple_temp:
                    dict_temp[key] = value
                GV.spike_out[gtask_id][target_gid][idx] = dict_temp
            self.pc_corr[gtask_id][target_gid][reference_gid] = [{'valid' : False, 'value' : None} for _ in range(window)]
    
    cpdef save_ss_elec(self, next_comp_lid, template_shift, timestep, index, debug_str, int ltask_id):
        electrode_type_id = 0
        gtask_id = GV.ltask_to_gtask[self.ind][ltask_id]
        gid = GV.lid_to_gid[gtask_id][self.ind][electrode_type_id][next_comp_lid]
        template_shift = Task.get_task(gtask_id).task_const['template_shift']
        GV.spike_out[gtask_id][gid].append(timestep - template_shift)

    cpdef save_ss_temp(self, next_comp_lid, template_shift, timestep, index, debug_str, int ltask_id):
        template_type_id = 1
        gtask_id = GV.ltask_to_gtask[self.ind][ltask_id]
        gid = GV.lid_to_gid[gtask_id][self.ind][template_type_id][next_comp_lid]
        template_shift = Task.get_task(gtask_id).task_const['template_shift']
        GV.spike_out[gtask_id][gid].append(timestep - template_shift)

    cpdef save_ss_temp_new(self, next_comp_lid, template_shift, timestep, index, debug_str, int ltask_id):
        template_type_id = 1
        gtask_id = GV.ltask_to_gtask[self.ind][ltask_id]
        gid = GV.lid_to_gid[gtask_id][self.ind][template_type_id][next_comp_lid]

    cpdef print(self, a, b, c, index, d, int task_id):
        print(a, b, c, d, task_id)
