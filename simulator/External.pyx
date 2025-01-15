import Task
cimport cython
import numpy as np
cimport numpy as np
from copy import deepcopy

import GlobalVars as GV
import copy
from libc.math cimport exp, tanh
import sys


cdef class ExternalModule:

    def __init__(self, Core.Core core, external_input, external_id):
        self.external_id = external_id
        self.core = core
        self.external_input = external_input
        self.idx = [0 for _ in range(len(self.external_input))]

        # pending ts & pending cyc
        self.pending_queue = []
        self.pending_cyc = []

        # For init operation
        self.init = True

        # Add an external core
        if external_id in GV.external_id:
            GV.external_modules.append(self)

        for gtask_id in range(GV.TOTAL_TASKS):
            idx = self.idx[gtask_id]
            while(idx < len(self.external_input[gtask_id])):
                if self.external_input[gtask_id][idx][0] < GV.initial_timestep[gtask_id]:
                    self.idx[gtask_id] += 1
                    idx = self.idx[gtask_id]
                else:
                    break
    
    cpdef pending_bci_send(self):
        for ind in range(len(self.pending_queue)):
            ltask_id, gtask_id = self.pending_queue[ind]
            cyc = self.pending_cyc[ind]
            if cyc <= GV.cyc:
                GV.valid_advance = True
                del self.pending_queue[ind]
                del self.pending_cyc[ind]
                # We should send the target_cycle
                target_cyc = GV.target_cyc[gtask_id]
                # Should start with init
                self.init = True
                self.core.isa_ctrl.init_bci_send(ltask_id, target_cyc)
                break
    
    # external_input is list [timestep, electrode_id, recording_dat]
    cpdef external_data(self, gtask_id):
        # Target CYC
        workload_cyc = GV.workload_cyc[gtask_id]
        timestep = GV.per_core_timestep[gtask_id][self.core.ind]
        assert((timestep * int(GV.sim_params['dt'][gtask_id] / GV.sim_params['cyc_period'])) == workload_cyc)
        # Target TS for the workload
        workload_ts = timestep % (GV.sim_params['workload_timestep'][gtask_id] + 1)
        # If the current cycle is below the workload cycle
        # => The external core simply waits
        if GV.cyc >= workload_cyc:
            # If not init => 
            if self.init:
                self.init = False
                return 'init', None

            idx = self.idx[gtask_id]
            bci_type_id = GV.sim_params['unit_type_name'][gtask_id].index('bci')
            bci_offset = 0
            for type_id in range(bci_type_id):
                bci_offset += GV.sim_params['num_unit'][gtask_id][type_id]

            while(idx < len(self.external_input[gtask_id])):
                # If the current cycle is above the workload cycle
                if self.external_input[gtask_id][idx][0] == workload_ts + GV.initial_timestep[gtask_id]:
                    external_dat = self.external_input[gtask_id][idx]
                    self.idx[gtask_id] += 1
                    idx = self.idx[gtask_id]
                    _, electrode_tid, recording_dat = external_dat
                    electrode_tid = int(electrode_tid)
                    # If the electrode is mapped on this core
                    if GV.tid_to_core[gtask_id][bci_type_id][electrode_tid][0] == self.core.ind:
                        electrode_gid = electrode_tid + bci_offset
                        return 'event', (GV.lid_to_gid[gtask_id][self.core.ind][bci_type_id].index(electrode_gid), recording_dat)
                else: break
            return 'no event', None
        else:
            assert(0)
