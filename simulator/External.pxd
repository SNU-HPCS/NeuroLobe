cimport Core
cimport Memory

cdef class ExternalModule:
    cdef Core.Core core
    cdef int external_id
    cdef list external_input
    cdef list idx

    cdef public int init

    cdef public list pending_queue
    cdef public list pending_cyc

    cpdef external_data(self, task_id)
    cpdef pending_bci_send(self)
