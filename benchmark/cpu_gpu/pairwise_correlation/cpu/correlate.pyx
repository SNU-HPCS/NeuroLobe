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


# Spatio-temporal iteration
cdef void accumulate_correlogram(unsigned int [:,:] sparse_conn_mat,
                                 unsigned int [:,:] spike_hist,
                                 unsigned int [:,:] correlogram,
                                 unsigned int [:] spike_list,
                                 unsigned int [:,:] conn_forward_ind,
                                 int num_spikes,
                                 int window,
                                 int t,
                                 int num_threads) noexcept nogil:

    cdef unsigned int corr
    cdef unsigned int spike

    cdef int neu_idx
    cdef int mem_addr
    cdef unsigned int dst_id
    cdef int src_id
    cdef unsigned int start
    cdef unsigned int end
    cdef int win

    # Searching and accumulating the history data to the partial c
    for neu_idx in range(num_spikes):
        src_id = spike_list[neu_idx]
        start = conn_forward_ind[src_id][0]
        end = conn_forward_ind[src_id][1]

        for mem_addr in prange(start, end, nogil=True, num_threads = num_threads):
            dst_id = sparse_conn_mat[mem_addr][1]
            for win in range(window):
                correlogram[mem_addr][win] += spike_hist[dst_id][(t - win) % window]

cdef void update_spike_history(int num_spikes,
                                unsigned int [:,:] spike_hist,
                                unsigned int [:] spike_list,
                                int t,
                                int window,
                                int cell_num):
    cdef int neu_idx
    cdef int src_id

    # Accumulate the spikes (cannot be parallelized)
    for neu_idx in range(num_spikes):
        src_id = spike_list[neu_idx]
        spike_hist[src_id][t % window] = spike_hist[src_id][t % window] + 1

cdef void reset_history(int cell_num,
                        unsigned int [:,:] spike_hist,
                        int t,
                        int window,
                        int num_threads
                        ) noexcept nogil:
    cdef int neu_idx

    for neu_idx in prange(cell_num, nogil = True, num_threads = num_threads):
        spike_hist[neu_idx][t % window] = 0

cdef void square_spikes(int cell_num,
                        unsigned int [:] spike_squared,
                        unsigned int [:,:] spike_hist,
                        int t,
                        int window,
                        int num_threads
                        ) noexcept nogil:
    cdef int neu_idx

    for neu_idx in prange(cell_num, nogil = True, num_threads = num_threads):
        spike_squared[neu_idx] += spike_hist[neu_idx][t % window] ** 2

cdef void calculate_pcc(int corr_period,
                        int t,
                        int window,
                        int conn_total,
                        unsigned int [:,:] sparse_conn_mat,
                        unsigned int [:] spike_squared,
                        unsigned int [:,:] correlogram,
                        int num_threads) noexcept nogil:
    
    cdef unsigned int curr_period
    cdef int conn_idx
    cdef int src_id
    cdef int dst_id
    cdef unsigned int src_spike
    cdef unsigned int dst_spike
    cdef float norm
    cdef int win
    cdef float pcc_ts

    if (t + 1) % corr_period == 0:
        curr_period = (t+1) // corr_period - 1

        for conn_idx in prange(conn_total, nogil=True, num_threads = num_threads):
            src_id = sparse_conn_mat[conn_idx][0]
            dst_id = sparse_conn_mat[conn_idx][1]
            src_spike = spike_squared[src_id]
            dst_spike = spike_squared[dst_id]
            norm = sqrt(src_spike * dst_spike)
            if norm > 0:
                for win in range(window):
                    #pcc[conn_idx, curr_period, win] = correlogram[conn_idx,win] / norm
                    pcc_ts = correlogram[conn_idx,win] / norm


def pairwise_correlation_cpu_inner(unsigned int [:,:] spike_hist,
                                   unsigned int [:] spike_squared,
                                   unsigned int [:] spike_list,
                                   unsigned int [:,:] conn_forward_ind,
                                   unsigned int [:,:] sparse_conn_mat, 
                                   unsigned int [:,:] correlogram,
                                   int cell_num,
                                   int num_spikes,
                                   int conn_total,
                                   int corr_period,
                                   int t,
                                   int window,
                                   int num_threads):

    # Update the spike history (shift + accum)
    # Reset history
    # post processing
    reset_history(cell_num, spike_hist, t, window, num_threads)
    
    # event-driven
    update_spike_history(num_spikes, spike_hist, spike_list, t, window, cell_num)

    # Square the spikes
    # check event
    square_spikes(cell_num, spike_squared, spike_hist, t, window, num_threads)
    
    # event-driven
    accumulate_correlogram(sparse_conn_mat, spike_hist, correlogram, spike_list,
                           conn_forward_ind, num_spikes, window, t, num_threads)

    # Calculate pcc
    # post processing
    calculate_pcc(corr_period, t, window, conn_total, sparse_conn_mat, spike_squared, correlogram, num_threads)


    return
