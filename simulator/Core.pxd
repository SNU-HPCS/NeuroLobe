from ISA import ISAController
from EventProcessor import PC
from External cimport ExternalModule
from Router cimport Router
from Memory cimport Memory
from Debug cimport DebugModule

cdef class Register:
    cdef int event_size
    cdef int num_tasks
    cdef int per_task_size
    cdef int free_region
    cdef int timestep_region
    cdef int general_region_size
    cdef int size
    cdef list reg
    cpdef read(self, addr)
    cpdef write(self, addr, dat)
    # cpdef print_dat(self, addr, task_id)

cdef class LoopCtrl:
    cdef object reg
    cdef int depth

    # num_iter: the number of iterations
    # count: the number of current iteration count
    cdef list num_iter
    cdef list counter
    cdef list timestep_addr

    cdef list forward_order

    cdef list inter_addr_offset
    cdef list intra_addr_offset

    # The instruction return address
    # (if the condition is not met (start_addr) / met (end_addr))
    cdef list inst_loop_addr
    cdef list inst_end_addr

    # the number of offset increment for each iteration
    cdef list loop_offset

    cpdef insert(self, func_type, outer_offset, inner_offset,
                 num_iter, count_addr, inst_loop_addr, inst_end_addr,
                 loop_offset, forward_order)
    cpdef iterate(self)
    cpdef get_addr(self, offset, width)

cdef class ProcessingEvent:
    cdef dict buf
    cdef object busy_type
    cdef object core
    cdef int ind
    cpdef busy(self)
    cpdef schedule_event(self, event_type, event_data, int task_id)
    cpdef event_done(self, event_type)
    cpdef get_data(self, event_type)
    cpdef set_end_state(self, event_type, end_cyc)
    cpdef print_buf(self)

cdef class Core:
    cdef public int ind
    cdef public int has_external
    cdef public int external

    cdef public Memory mem
    cdef public ExternalModule external_module
    cdef public Router router
    cdef public DebugModule debug_module
    cdef public Register reg
    cdef public LoopCtrl loop_ctrl

    cdef public object isa_ctrl
    cdef object pc
    cdef public object processingEvent

    #cdef public list inst_graph
    
    cdef public list pending_queue

    cpdef core_advance(self)
    cpdef process_events(self)
    cpdef process_external(self)
    cpdef schedule_events(self)
