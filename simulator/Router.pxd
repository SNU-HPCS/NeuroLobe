cimport Core
cimport Memory
cimport NoC
import numpy as np
cimport numpy as np
ctypedef np.uint8_t uint8

cdef class TaskScheduler:
    cdef int ind
    cdef Router router
    cdef list latency_budget
    cdef list sorted_task

    # This is done every 1 ms
    cpdef task_init(self, task_id, budget)
    cpdef decrement_budget(self)
    cpdef schedule(self, valid)

cdef class PriorityBuffer:
    cdef int ind
    cdef list buf
    cdef list valid
    cdef TaskScheduler task_scheduler

    cpdef empty(self)
    cpdef append(self, task_id, event_type, event)
    cpdef popleft(self)
    cpdef get_buf(self)

cdef class Synchronizer:
    cdef Router router
    cdef Core.Core core
    cdef Memory.Memory mem

    cdef int ind

    # This should be a dictionary for each packet type
    cdef int sync_target
    cdef public dict sync_table

    cpdef event_done(self, event_type, pck_type, event_data, task_id)
    
    cpdef increment_ack(self, pck_type, task_id)
    #cpdef increment_ack(self, pck_type, task_id, ack_addr)
    #cpdef decrement_ack(self, pck_type, ack_addr, task_id)
    cpdef decrement_ack(self, depth, ack_num, task_id)
    cpdef synchronize(self)

cdef class Router:
    # position
    cdef int ind
    cdef int x
    cdef int y
    cdef int external
    cdef Core.Core core

    # Routing
    cdef public object recv_buf
    cdef public list send_buf
    cdef public object in_event_buf
    cdef public object out_event_buf
    cdef public long long generator_available_cyc
    cdef public long long receiver_available_cyc
    cdef public tuple receiver_dat
    cdef public NoC.Switch sw

    cdef public Synchronizer synchronizer
    cdef public TaskScheduler task_scheduler

    cdef public list packet_to_event
    cdef public list packet_to_route
    cdef public list task_translation

    # Core
    cdef Memory.Memory mem

    cpdef get_packet_to_event(self, mode, pck_type, task_id)
    cpdef get_packet_to_route(self, pck_type, task_id)
    cpdef packet_generation(self)
    cpdef packet_reception(self)
