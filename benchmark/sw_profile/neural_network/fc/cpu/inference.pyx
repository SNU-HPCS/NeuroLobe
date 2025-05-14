# cython: profile=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
# cython: boundscheck=False

import cython
import numpy as np
cimport numpy as np
from cython.parallel import prange
from cython.parallel cimport parallel
from cython import boundscheck, wraparound
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf

from libc.math cimport sqrt
cimport openmp

cdef void accumulate_hid_spikes(float [:] hidden_vector,
                             float [:] conn_mem_2_weight_transpose,
                             float [:] I_t_vector2,
                             float [:] output_vector,
                             float [:] weight_list_2,
                             int h_dim,
                             int o_dim,
                             int t,
                             int num_threads) noexcept nogil:
                             # int num_threads):
    cdef int src_id
    cdef int dst_id

    cdef float weight
    cdef float spike_value

    cdef float weight_temp
    cdef int idx_temp

    # for dst_id in range(o_dim):
    for dst_id in prange(o_dim, nogil=True, num_threads = num_threads):
        weight_temp = 0
        idx_temp = dst_id * h_dim
        for src_id in range(h_dim):
            spike_value = hidden_vector[src_id]

            if spike_value > 0:
                weight = conn_mem_2_weight_transpose[idx_temp+src_id]
                weight_temp = weight_temp + weight * spike_value
        weight_list_2[dst_id] = weight_temp


cdef void accumulate_ext_spikes( int num_spikes,
                             float [:] spike_sum,
                             float [:] spike_weight,
                             float [:] conn_mem_1_weight_transpose,
                             float [:] I_t_vector1,
                             float [:] hidden_vector,
                             float [:] weight_list_1,
                             int i_dim,
                             int h_dim,
                             int t,
                             int num_threads) noexcept nogil:
    cdef int src_neu_idx
    cdef int src_id
    cdef int dst_id

    cdef float weight
    cdef float spike_value

    cdef float weight_temp
    cdef int idx_temp

    # for dst_id in range(h_dim):
    for dst_id in prange(h_dim, nogil=True, num_threads = num_threads):
        weight_temp = 0
        idx_temp = dst_id * i_dim
        for src_id in range(i_dim):
            weight = conn_mem_1_weight_transpose[idx_temp+src_id]
            weight_temp = weight_temp + weight * spike_sum[src_id]
        weight_list_1[dst_id] = weight_temp

cdef void update_output(int o_dim,
                              int num_threads,
                              float [:] output_vector,
                              float [:] weight_list_2,
                              float [:] I_t_vector2
                              ) noexcept nogil:
    cdef int dst_id

    for dst_id in prange(o_dim, nogil=True, num_threads = num_threads):
        output_vector[dst_id] = (weight_list_2[dst_id] + I_t_vector2[dst_id])

cdef void update_hidden(int h_dim,
                        int num_threads,
                        float [:] hidden_vector,
                        float [:] weight_list_1,
                        float [:] I_t_vector1
                        ) noexcept nogil:
    cdef int dst_id

    for dst_id in prange(h_dim, nogil=True, num_threads = num_threads):
        hidden_vector[dst_id] = (weight_list_1[dst_id] + I_t_vector1[dst_id])

cdef void bin_ext_spikes(int [:] spike_list,
                         float [:] spike_weight,
                         float [:,:] spike_buf,
                         float [:] spike_sum,
                         int h_dim,
                         int bin_width,
                         int num_spikes,
                         int t,
                         int num_threads
                         ) noexcept nogil:
    
    cdef int src_neu_idx
    cdef int src_id
    cdef int dst_id

    cdef float spike_value

    for src_neu_idx in range(num_spikes):
        src_id = spike_list[src_neu_idx]
        spike_value = spike_weight[src_neu_idx]
        spike_buf[src_id, t % bin_width] += spike_value
        spike_sum[src_id] += spike_value

cdef void reset_spike_buf(float [:,:] spike_buf,
                          float [:] spike_sum,
                          int i_dim,
                          int h_dim,
                          int bin_width,
                          int t,
                          int num_threads
                          ) noexcept nogil:
    cdef int src_id

    for src_id in prange(i_dim, nogil=True, num_threads = num_threads):
        spike_sum[src_id] = spike_sum[src_id] - spike_buf[src_id,(t) % bin_width]
        spike_buf[src_id, (t) % bin_width] = 0
            
def FC_cpu_inner(         int [:] spike_list, float [:] spike_weight,
                                   float [:] conn_mem_1_weight_transpose,
                                   float [:] conn_mem_2_weight_transpose,
                                   float [:] hidden_vector,
                                   float [:] output_vector,
                                   float [:] weight_list_1,
                                   float [:] weight_list_2,
                                   float [:] I_t_vector1,
                                   float [:] I_t_vector2,
                                   float [:,:] spike_buf,
                                   float [:] spike_sum,
                                   int i_dim,
                                   int h_dim,
                                   int o_dim,
                                   int t,
                                   int num_spikes,
                                   int num_threads,
                                   int bin_width):
    # hidden layer to output layer 
    # event-driven
    if t > 0: 
        accumulate_hid_spikes(hidden_vector, conn_mem_2_weight_transpose, I_t_vector2, output_vector, weight_list_2, h_dim, o_dim, t, num_threads)
    
        #check event
        update_output(o_dim, num_threads, output_vector, weight_list_2, I_t_vector2)
    
    # reset spike_buf
    # post processing
    reset_spike_buf(spike_buf, spike_sum, i_dim, h_dim, bin_width, t, num_threads)

    # dynamically bin spikes
    # event-driven
    bin_ext_spikes(spike_list, spike_weight, spike_buf, spike_sum, h_dim, bin_width, num_spikes, t, num_threads)

    # input layer to hidden layer
    # event-driven
    accumulate_ext_spikes(num_spikes, spike_sum, spike_weight, conn_mem_1_weight_transpose, I_t_vector1, hidden_vector, weight_list_1, i_dim, h_dim, t, num_threads)
    
    # check event
    update_hidden(h_dim, num_threads, hidden_vector, weight_list_1, I_t_vector1)
    
    return
