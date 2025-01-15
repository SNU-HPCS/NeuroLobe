cimport Core

cdef class DebugModule:
    cdef public Core.Core core
    cdef public int ind

    # for debugging pc
    cdef public list pc_corr

    # TM
    cpdef save_tm_corr(self, int tm, x, int ind, str_debug, int task_id)

    # SNN
    cpdef save_snn_spike(self, int spiked, int timestep, int lid, str_debug, int task_id)
    cpdef save_snn_state(self, float state, int timestep, int lid, str_debug, int task_id)

    # PC
    cpdef save_pc_corr(self, int target_lid, int reference_pid, x, str_debug, int task_id)

    # SS
    cpdef save_ss_elec(self, next_comp_lid, template_shift, timestep, debug_str, int task_id)
    cpdef save_ss_temp(self, next_comp_lid, template_shift, timestep, debug_str, int task_id)
