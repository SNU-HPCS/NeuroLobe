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

cdef void vector_pcc(float [:] c1,
                     float [:] c2,
                     float [:] c3,
                     unsigned long [:] S1,
                     unsigned int [:] S2,
                     unsigned int [:] S3,
                     float [:,:] corr,
                     int t_shift,
                     int num_templates,
                     int num_threads) noexcept nogil:

    cdef float t1
    cdef float t2
    cdef float t3

    cdef int temp_idx

    if t_shift > 0:
        for temp_idx in prange(num_templates, nogil=True, num_threads = num_threads):
            t1 = (c1[temp_idx] * S1[temp_idx] - c2[temp_idx] * S2[0])
            t1 = t1 ** 2
            t2 = c1[temp_idx] * S3[0]
            t3 = S2[0] ** 2
            t2 = t2 - t3
            t2 = c3[temp_idx] * t2
            t1 = t1 / t2
            corr[temp_idx,t_shift] = t1

cdef void calc_S23(unsigned int [:,:] spike_hist,
                   unsigned int [:] S2,
                   unsigned int [:] S3,
                   unsigned int t,
                   unsigned int num_templates,
                   unsigned int cell_num,
                   unsigned int template_width,
                   unsigned int num_threads) noexcept nogil:

    cdef unsigned int n
    cdef unsigned int temp_idx

    cdef unsigned int new_spike
    cdef unsigned int prev_spike


    cdef unsigned int S2_temp = 0;
    cdef unsigned int S3_temp = 0;
    for n in prange(cell_num, nogil=True, num_threads = num_threads):
        new_spike = spike_hist[n, t % (template_width + 1)]
        prev_spike = spike_hist[n, (t+1) % (template_width + 1)]
        S2_temp += new_spike - prev_spike
        S3_temp += new_spike**2 - prev_spike ** 2
    S2[0] += S2_temp
    S3[0] += S3_temp

cdef void reset_S1(int num_templates,
                   unsigned long [:] S1,
                   unsigned long [:,:] S1_partial_hist,
                   int t,
                   int template_width
                   ):
    cdef int temp_idx

    for temp_idx in range(num_templates):
        S1[temp_idx] -= S1_partial_hist[temp_idx][(t + template_width - 1) % (template_width)]
        S1_partial_hist[temp_idx][(t + template_width - 1) % (template_width)] = 0


cdef void accumulate_S1_hist(unsigned long [:] S1,
                             unsigned int [:,:] spike_hist,
                             unsigned long [:,:] S1_partial_hist,
                             unsigned int [:,:,:] temp_mat,
                             unsigned int t,
                             unsigned int num_templates,
                             unsigned int cell_num,
                             unsigned int template_width,
                             unsigned int num_threads) noexcept nogil:

    cdef unsigned int n
    cdef unsigned int temp_idx
    cdef unsigned int t_shift

    cdef unsigned int template
    cdef unsigned int spike


    for n in range(cell_num):
        spike = spike_hist[n, t % (template_width + 1)]
        if spike > 0:
            for t_shift in prange(template_width, nogil=True, num_threads = num_threads):
                for temp_idx in range(num_templates):
                    template = temp_mat[temp_idx,n,template_width - t_shift - 1]
                    S1_partial_hist[temp_idx][(t + t_shift) % (template_width)] += template * spike

cdef void calc_S1(int num_templates,
                          unsigned long [:] S1,
                          unsigned long [:,:] S1_partial_hist,
                          int t,
                          int template_width):
    cdef int temp_idx

    for temp_idx in range(num_templates):
        S1[temp_idx] += S1_partial_hist[temp_idx][t % (template_width)]

cdef void update_spike_history(unsigned int [:] spike_list,
                               unsigned int [:,:] spike_hist,
                               int t,
                               int template_width):
    cdef int neu_idx
    cdef int src_id
    
    num_spikes = len(spike_list)
    for neu_idx in range(num_spikes):
        src_id = spike_list[neu_idx]
        spike_hist[src_id][t % (template_width + 1)] += 1

    return                           

cdef void reset_history(int cell_num,
                        int num_threads,
                        unsigned int [:,:] spike_hist,
                        int t,
                        int template_width,
                        ) noexcept nogil:
    cdef unsigned int n
    
    for n in prange(cell_num, nogil=True, num_threads = num_threads):
        spike_hist[n, t % (template_width + 1)] = 0
    

def template_matching_cpu_sparse_inner(unsigned int [:] spike_list,
                                       unsigned int [:,:] spike_hist,
                                       unsigned long [:,:] S1_partial_hist,
                                       unsigned int [:,:,:] temp_mat,
                                       float [:] c1,
                                       float [:] c2,
                                       float [:] c3,
                                       unsigned long [:] S1,
                                       unsigned int [:] S2,
                                       unsigned int [:] S3,
                                       float [:,:] corr,
                                       unsigned int t,
                                       unsigned int num_templates,
                                       unsigned int cell_num,
                                       unsigned int bin_width,
                                       unsigned int template_width,
                                       unsigned int num_threads):

    # dynamically bin the spikes here
    # Reset history
    # check event
    reset_history(cell_num, num_threads, spike_hist, t, template_width)
    
    # event-driven
    update_spike_history(spike_list, spike_hist, t, template_width)

    # check event
    reset_S1(num_templates, S1, S1_partial_hist, t, template_width)

    # event-driven
    accumulate_S1_hist(S1, spike_hist, S1_partial_hist, temp_mat, t, num_templates, cell_num, template_width, num_threads)

    # post processing
    calc_S1(num_templates, S1, S1_partial_hist, t, template_width)

    # event-driven 
    calc_S23(spike_hist, S2, S3, t, num_templates, cell_num, template_width, num_threads)
    
    # post processing
    vector_pcc(c1, c2, c3, S1, S2, S3, corr, t - template_width + 1, num_templates, num_threads)

