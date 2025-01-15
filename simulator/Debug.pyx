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

        self.pc_corr = [[] for _ in range(GV.TOTAL_TASKS)]

    cpdef save_tm_corr(self, int tm, x, int ind, str_debug, int ltask_id):
        template_type_id = 1
        gtask_id = GV.ltask_to_gtask[self.ind][ltask_id]
        temp_gid = GV.lid_to_gid[gtask_id][self.ind][template_type_id][tm]
        if not np.isnan(x):
            GV.spike_out[gtask_id][temp_gid].append([x, ind]) # append PCC value

    cpdef save_snn_spike(self, int spiked, int timestep, int lid, str_debug, int ltask_id):
        neuron_type_id = 0
        gtask_id = GV.ltask_to_gtask[self.ind][ltask_id]
        gid = GV.lid_to_gid[gtask_id][self.ind][neuron_type_id][lid]
        if spiked != 0: GV.spike_out[gtask_id][gid].append(timestep)

    cpdef save_snn_state(self, float state, int timestep, int lid, str_debug, int ltask_id):
        neuron_type_id = 0
        gtask_id = GV.ltask_to_gtask[self.ind][ltask_id]
        gid = GV.lid_to_gid[gtask_id][self.ind][neuron_type_id][lid]

        # if gid in GV.debug_dst:
        GV.debug_list['state'][gtask_id].write("{}ts: {}, neu: {}, v_t: {:.4f}\n".format(str_debug, timestep, gid, state))
        
    cpdef save_pc_corr(self, int target_lid, int reference_pid, x, str_debug, int ltask_id):
        # This is for debugging
        # Save the correlation result to the external memory
        neuron_type_id = 0
        gtask_id = GV.ltask_to_gtask[self.ind][ltask_id]
        target_gid = GV.lid_to_gid[gtask_id][self.ind][neuron_type_id][target_lid]
        reference_gid = GV.pid_to_gid[gtask_id][self.ind][neuron_type_id][neuron_type_id][reference_pid]
        # timestep = self.core.reg.read(19, gtask_id)
        timestep = GV.per_core_timestep[gtask_id][self.ind]

        idx = timestep // GV.sim_params['consts'][gtask_id]['corr_period']
       
        self.pc_corr[gtask_id].append(x)

        if len(self.pc_corr[gtask_id]) == GV.sim_params['consts'][gtask_id]['window'] + 1:
            while len(GV.spike_out[gtask_id][target_gid]) <= idx:
                GV.spike_out[gtask_id][target_gid].append({})

            if sum(self.pc_corr[gtask_id]):
                # if reference_gid not in GV.spike_out[gtask_id][target_gid][idx].keys():
                #     GV.spike_out[gtask_id][target_gid][idx][reference_gid] = []
                GV.spike_out[gtask_id][target_gid][idx][reference_gid] = self.pc_corr[gtask_id]
                dict_temp = {}
                tuple_temp = sorted(GV.spike_out[gtask_id][target_gid][idx].items())
                for key, value in tuple_temp:
                    dict_temp[key] = value
                GV.spike_out[gtask_id][target_gid][idx] = dict_temp
            self.pc_corr[gtask_id] = []
    
    cpdef save_ss_elec(self, next_comp_lid, template_shift, timestep, debug_str, int ltask_id):
        electrode_type_id = 0
        gtask_id = GV.ltask_to_gtask[self.ind][ltask_id]
        gid = GV.lid_to_gid[gtask_id][self.ind][electrode_type_id][next_comp_lid]
        template_shift = Task.get_task(gtask_id).task_const['template_shift']
        GV.spike_out[gtask_id][gid].append(timestep - template_shift)

    cpdef save_ss_temp(self, next_comp_lid, template_shift, timestep, debug_str, int ltask_id):
        template_type_id = 1
        gtask_id = GV.ltask_to_gtask[self.ind][ltask_id]
        gid = GV.lid_to_gid[gtask_id][self.ind][template_type_id][next_comp_lid]
        template_shift = Task.get_task(gtask_id).task_const['template_shift']
        GV.spike_out[gtask_id][gid].append(timestep - template_shift)

