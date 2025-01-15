cimport Core

cdef class Memory:
    cdef Core.Core core
    cdef int ind

    cdef list neu_num
    cdef list pre_neu_num

    cdef public list state_mem
    cdef public list state_mem_offset
    cdef int state_mem_cache_size
    cdef list state_mem_cache_dat
    cdef list state_mem_cache_addr
    cdef list state_mem_cache_dirty
    cdef list state_mem_cache_queue

    # Three memory to keep data
    cdef list hist_mem_num_entries
    cdef public list hist_mem_offset
    cdef list hist_mem
    cdef public list hist_metadata
    #cdef public list hist_unit_offset
    #cdef public list hist_log2_precision
    #cdef public list hist_length

    cdef list hist_task_translator
    # two temporay dataset to enable 
    # an RMW operation within a single cycle
    cdef int hist_mem_cache_size
    cdef list hist_pos
    cdef list hist_mem_cache_dat
    cdef list hist_mem_cache_addr
    cdef list hist_mem_cache_dirty
    cdef list hist_mem_cache_queue
    
    #
    #cdef list stack_mem_num_entries
    #cdef public list stack_mem_offset
    #cdef list stack_mem
    #cdef list stack_ptr
    
    #
    cdef list corr_mem_num_entries
    cdef list corr_mem
    cdef list corr_mem_offset
    cdef int corr_mem_cache_size
    cdef list corr_mem_cache_dat
    cdef list corr_mem_cache_addr
    cdef list corr_mem_cache_dirty
    cdef list corr_mem_cache_queue

    #
    cdef list corr_forward_num_entries
    cdef public list corr_forward_offset
    cdef list corr_forward

    # Routing table
    cdef list route_forward_num_entries
    cdef public list route_forward_offset
    cdef list route_forward

    cdef list route_mem_num_entries
    cdef list route_mem
    cdef list route_mem_offset

    #
    cdef list ack_left_mem_num_entries
    cdef public list ack_left_mem_offset
    cdef list ack_left_mem

    #
    cdef list ack_num_mem_num_entries
    cdef public list ack_num_mem_offset
    cdef list ack_num_mem

    cdef list ack_stack_mem_num_entries
    cdef public list ack_stack_mem_offset
    cdef list ack_stack_mem
    cdef list ack_stack_ptr

    # cpdef read_corr_inverse_indirect(self, data_type, task_id, virtual_addr)
    # cpdef read_corr_inverse_mem(self, data_type, task_id, virtual_addr)
    cpdef read_corr(self, data_type, task_id, virtual_addr, offset)
    cpdef write_corr(self, data_type, task_id, virtual_addr, write_dat, offset)
    cpdef accum_corr_mem(self, func_type, data_type, ltask_id, virtual_addr, offset, write_dat, alu_dat = *)
    cpdef read_corr_forward(self, data_type, mem_offset, task_id, virtual_addr)

    cpdef read_state_mem(self, data_type, mem_offset, task_id, virtual_addr)
    cpdef write_state_mem(self, data_type, mem_offset, task_id, virtual_addr, write_dat)
    cpdef accum_state_mem(self, func_type, data_type, mem_offset, task_id, virtual_addr, write_dat, alu_dat = *)
    
    cpdef read_hist_mem(self, data_type, mem_offset, task_id, virtual_addr, pos)
    cpdef write_hist_mem(self, data_type, mem_offset, task_id, virtual_addr, pos, write_dat)
    cpdef accum_hist_mem(self, func_type, data_type, mem_offset, task_id, virtual_addr, pos, write_dat, alu_dat = *)
    cpdef increment_pos(self, ltask_id)
    cpdef get_hist_addr(self, data_type, ltask_id, virtual_addr, pos)

    #cpdef push_stack_mem(self, data_type, mem_offset, task_id, init_write_dat)
    #cpdef pop_stack_mem(self, data_type, mem_offset, task_id)
    #cpdef check_empty(self, data_type, mem_offset, task_id)
    
    cpdef read_route_forward(self, data_type, task_id, virtual_addr)
    cpdef read_route(self, data_type, task_id, virtual_addr)

    cpdef read_ack_left(self, data_type, task_id, virtual_addr)
    cpdef write_ack_left(self, data_type, task_id, virtual_addr, write_dat)

    cpdef read_ack_num(self, data_type, task_id, virtual_addr)
    cpdef write_ack_num(self, data_type, task_id, virtual_addr, write_dat)
    
    cpdef push_ack_stack_mem(self, data_type, task_id, init_write_dat)
    cpdef pop_ack_stack_mem(self, data_type, task_id)

    # cpdef memory_writeback(self, use_corr, use_hist, use_state)

    cpdef save_mem(self, arch_state)