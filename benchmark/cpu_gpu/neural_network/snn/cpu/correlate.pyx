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

cdef void accumulate_ext_spikes( int num_spikes,
                             int [:] spike_list,
                             int [:] conn_mem_1_delay,
                             float [:] conn_mem_1_weight,
                             float [:] hidden_vector,
                             float [:] I_t_vector1,
                             float [:] g_t_vector1,
                             float [:] decay_v_vector1,
                             float [:] decay_g_vector1,
                             float [:,:] weight_accum1,
                             int input_dim,
                             int h_dim,
                             int t,
                             int window,
                             int num_threads) noexcept nogil:
    cdef int neu_idx
    cdef int src_id
    cdef int dst_id

    cdef int delay
    cdef float weight

    cdef int idx_temp
    cdef int conn_mem_idx

   # accumulate weight to hidden vector 
    for dst_id in prange(h_dim, nogil=True, num_threads = num_threads):
        idx_temp = dst_id * input_dim
        for neu_idx in range(num_spikes):
            src_id = spike_list[neu_idx]
            conn_mem_idx = idx_temp + src_id

            delay = conn_mem_1_delay[conn_mem_idx]
            weight = conn_mem_1_weight[conn_mem_idx]

            weight_accum1[dst_id][(t+delay) % window] += weight


cdef void accumulate_neu_spikes( float [:] hidden_vector,
                                 int [:] conn_mem_2_delay,
                                 float [:] conn_mem_2_weight,
                                 float [:] output_vector,
                                 float [:] I_t_vector2,
                                 float [:] g_t_vector2,
                                 float [:] decay_v_vector2,
                                 float [:] decay_g_vector2,
                                 float [:,:] weight_accum2,
                                 float [:] threshold,
                                 int [:] hidden_spiked_list,
                                 int h_dim,
                                 int o_dim,
                                 int t,
                                 int window,
                                 int num_threads) noexcept nogil:
    cdef int neu_idx
    cdef int dst_id
    cdef int src_id

    cdef int delay
    cdef float weight

    cdef int idx_temp
    cdef int conn_mem_idx
   

    # accumulate weight to output vector
    # if t > 0:
    for dst_id in prange(o_dim, nogil=True, num_threads = num_threads):
        idx_temp = dst_id * h_dim
        for src_id in range(h_dim):
            if hidden_vector[src_id] > threshold[src_id]:
                hidden_spiked_list[src_id] = 1 # mark as spiked 

                conn_mem_idx = idx_temp + src_id

                delay = conn_mem_2_delay[conn_mem_idx]
                weight = conn_mem_2_weight[conn_mem_idx]

                weight_accum2[dst_id][(t+delay) % window] += weight

cdef void update_hid_state(int h_dim,
                           int num_threads,
                           int t,
                           int window,
                           float [:,:] weight_accum1,
                           float [:] I_t_vector1,
                           float [:] g_t_vector1,
                           float [:] decay_v_vector1,
                           float [:] decay_g_vector1,
                           float [:] hidden_vector,
                           ) noexcept nogil:
    cdef int neu_idx
    cdef float weight_state
    cdef float g_t_state
    cdef float v_t_state

    for neu_idx in prange(h_dim, nogil=True, num_threads = num_threads):
        # pop weight accum
        weight_state = weight_accum1[neu_idx][t % window]
        weight_accum1[neu_idx][t % window] = 0
        
        if t > 0:
            weight_state = weight_state + I_t_vector1[neu_idx]
        g_t_state = g_t_vector1[neu_idx] + weight_state
        if t > 0:
            v_t_state = decay_v_vector1[neu_idx] * hidden_vector[neu_idx]
            v_t_state = v_t_state + g_t_state
            hidden_vector[neu_idx] = v_t_state
        g_t_vector1[neu_idx] = g_t_state * decay_g_vector1[neu_idx]

cdef void reset_hid(int h_dim,
                    int [:] hidden_spiked_list,
                    float [:] hidden_vector,
                    int num_threads
                    ) noexcept nogil:
    cdef int neu_idx

    for neu_idx in prange(h_dim, nogil=True, num_threads = num_threads):
        if hidden_spiked_list[neu_idx]: # initialize spiked neurons
            hidden_spiked_list[neu_idx] = 0 
            hidden_vector[neu_idx] = 0

cdef void update_neu_state(int o_dim,
                           int num_threads,
                           int t,
                           int window,
                           float [:,:] weight_accum2,
                           float [:] I_t_vector2,
                           float [:] g_t_vector2,
                           float [:] decay_v_vector2,
                           float [:] decay_g_vector2,
                           float [:] output_vector
                           ) noexcept nogil:
    cdef int neu_idx
    cdef float weight_state
    cdef float g_t_state
    cdef float v_t_state

    for neu_idx in prange(o_dim, nogil=True, num_threads = num_threads):
        # pop weight accum
        weight_state = weight_accum2[neu_idx][t % window]
        weight_accum2[neu_idx][t % window] = 0

        if t > 1:
            weight_state = weight_state + I_t_vector2[neu_idx]
        g_t_state = g_t_vector2[neu_idx] + weight_state
        
        if t > 1:
            v_t_state = decay_v_vector2[neu_idx] * output_vector[neu_idx]
            v_t_state = v_t_state + g_t_state
            output_vector[neu_idx] = v_t_state
        g_t_vector2[neu_idx] = g_t_state * decay_g_vector2[neu_idx]


def SlayerFCSNN_cpu_inner(         int [:] spike_list,
                                   int [:] conn_mem_1_delay,
                                   float [:] conn_mem_1_weight,
                                   int [:] conn_mem_2_delay,
                                   float [:] conn_mem_2_weight,
                                   float [:,:] weight_accum1,
                                   float [:,:] weight_accum2,
                                   float [:] I_t_vector1,
                                   float [:] I_t_vector2,
                                   float [:] hidden_vector,
                                   float [:] output_vector,
                                   float [:] decay_v_vector1,
                                   float [:] decay_v_vector2,
                                   float [:] g_t_vector1,
                                   float [:] g_t_vector2,
                                   float [:] decay_g_vector1,
                                   float [:] decay_g_vector2,
                                   float [:] threshold,
                                   int [:] hidden_spiked_list,
                                   int input_dim,
                                   int h_dim,
                                   int o_dim,
                                   int t,
                                   int window,
                                   int num_spikes,
                                   int num_threads):

    # input layer to hidden layer
    # event-driven
    accumulate_ext_spikes(num_spikes, spike_list, conn_mem_1_delay, conn_mem_1_weight, hidden_vector,I_t_vector1, g_t_vector1, decay_v_vector1, decay_g_vector1, weight_accum1, input_dim, h_dim, t, window, num_threads)

    # update hidden neuron state
    # check event
    update_hid_state(h_dim, num_threads, t, window, weight_accum1, I_t_vector1, g_t_vector1, decay_v_vector1, decay_g_vector1, hidden_vector)

    # hidden layer to output layer
    # event-driven
    accumulate_neu_spikes(hidden_vector, conn_mem_2_delay, conn_mem_2_weight, output_vector, I_t_vector2, g_t_vector2, decay_v_vector2, decay_g_vector2, weight_accum2, threshold, hidden_spiked_list, h_dim, o_dim, t, window, num_threads)

    # initialize hidden vector state
    # post processing
    reset_hid(h_dim, hidden_spiked_list, hidden_vector, num_threads)
    
    # update state with accumulated weights
    # check event
    update_neu_state(o_dim, num_threads, t, window, weight_accum2, I_t_vector2, g_t_vector2, decay_v_vector2, decay_g_vector2, output_vector)
    
    return
