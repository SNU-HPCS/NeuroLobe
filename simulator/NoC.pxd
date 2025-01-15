cimport Router

cpdef enum DirectionIndex:
    left = 0
    bot = 1
    top = 2
    right = 3
    direction_max = 4

cdef class Switch:
    cdef public int cyc
    cdef public int external
    cdef public int is_active
    cdef public int flag_deact
    cdef public int in_activation_list
    cdef public int in_deactivation_list
    cdef public int x
    cdef public int y
    cdef public Router.Router router
    cdef public list is_edge
    cdef public object switches
    cdef public object delay
    cdef public object i_data
    cdef public object o_data
    cdef public object o_data_next
    cdef public object o_buf
    cdef public object buf_pos

    cpdef put_buf(self, pck)
    cpdef put_activation_list(self, switch, activation_list)
    cpdef put_deactivation_list(self, switch, deactivation_list)
    cpdef advance(self, activation_list)
    cpdef calc_next(self, deactivation_list)


#cdef class MergeSplit:
#    cdef public int cyc
#    cdef public int is_active
#    cdef public int flag_deact
#    cdef public int in_activation_list
#    cdef public int in_deactivation_list
#    cdef public int direction
#    cdef public int is_horizontal
#    cdef public int length
#    cdef public list merging_switches
#    cdef public object neighbor_block
#    cdef public int delay
#    cdef public int chip_delay
#    cdef public object i_data
#    cdef public object o_data_next
#    cdef public object o_buf
#    cdef public object buf_pos
#    cdef public object chip_i_data
#    cdef public object chip_o_data
#    cdef public object chip_o_data_next
#    cdef public object chip_o_buf
#    cdef public object chip_latency_buf
#    cdef public object off_chip_window
#
#    cpdef put_buf(self, pck, packet_pos)
#    cpdef put_activation_list(self, block, activation_list)
#    cpdef advance(self, activation_list)
#    cpdef calc_next(self, deactivation_list)

cdef class NoC:
    cdef public int cyc
    cdef public int max_x
    cdef public int max_y
    cdef public object merge_split
    cdef public object merge_split_external
    cdef public object active_sw_list
    cdef public object sw
    cdef public object sw_external

    cpdef noc_advance(self)
    cpdef send_pck_to_sw(self, router)
    cpdef activate_sw(self, switch)
    cpdef deactivate_sw(self, switch)
