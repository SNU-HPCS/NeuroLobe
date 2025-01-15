# cython: boundscheck=False

cimport cython
cimport numpy as np
import numpy as np

cpdef init_spike_sync_topology(list route_forward, list route_mem, list cores, int external)
cpdef load_external_stimulus(external_filename)
cpdef init_hist_mem(initial_hist_mem, max_delay, name, type_id, initial_state, task_id)
cpdef init_ack_left_mem(initial_ack_left_mem, name, task_id)
cpdef init_ack_num_mem(initial_ack_num_mem, name, task_id)
cpdef init_ack_stack_mem(initial_ack_stack_mem, name, task_id)
cpdef init_corr_mem(conn_filename, mapping_filename, cores, task_id)
cpdef init_other_mem(state_filename, task_info, task_id)
cpdef init_task_mem(core_list)
cpdef allocate_entry(entry, mem, offset, align_info)
cpdef ceil_div(a, b)
cpdef mapping(list gid_to_core, list tid_to_core, list lid_to_gid, mapping_filename, int task_id, conn_list, p_temp_table, p_neu_table)
cpdef init_network (network_filename, task_id)
