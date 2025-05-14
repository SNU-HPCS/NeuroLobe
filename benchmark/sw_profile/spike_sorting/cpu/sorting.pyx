# cython: profile=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
# cython: boundscheck=False

import cython
import numpy as np
cimport numpy as np
from cython.parallel import prange
from cython.parallel cimport parallel
from libc.math cimport sqrt
cimport openmp
from libc.stdio cimport printf

cdef void subtract_overlap(unsigned int [:] matched_temp_idx_list,
                           int n_tm,
                           int n_e,
                           int n_t,
                           int curr_idx,
                           int template_shift,
                           int max_conn_elec,
                           float [:,:] bci_fifo,
                           float [:] best_amp_list, 
                           float [:,:,:] templates,
                           unsigned int [:,:] temp_to_elec,
                           float [:] similarity,
                           int window,
                           int num_threads,
                           unsigned int [:] total_valid_number,
                           unsigned int [:] result_spiketimes, 
                           float [:] result_amplitudes) noexcept nogil:
    cdef unsigned int temp
    cdef int elec_idx
    cdef int elec
    cdef int t
    cdef unsigned int count = 0;
    cdef unsigned int total_idx = 0;

    # for temp in range(n_tm):
    for temp in prange(n_tm, nogil=True, num_threads=num_threads):
        if matched_temp_idx_list[temp]:
            # subtract overlap for ALL connected template_indices
            for elec_idx in range(max_conn_elec):
                elec = temp_to_elec[temp][elec_idx]
                if elec == n_e:
                    break
                for t in range(n_t):
                    bci_fifo[elec][(curr_idx-n_t+t+1)%window] = bci_fifo[elec][(curr_idx-n_t+t+1)%window] + best_amp_list[temp] * templates[temp][elec][t]
            matched_temp_idx_list[temp] = 0
            # FOR DEBUGGING
            # count += 1
            # total_idx = total_valid_number[0]+count-1
            # printf("Template: %d, Timestep: %d\n",temp,curr_idx-template_shift)
            # result_spiketimes[temp] = curr_idx-template_shift
            #result_amplitudes[total_idx] = best_amp_n[temp]
    # total_valid_number[0] += count
    return

cdef void calc_similarity(int max_conn_temp,
                      int max_conn_elec,
                      unsigned int [:,:] elec_to_temp,
                      unsigned int [:,:] temp_to_elec,
                      int n_tm,
                      int n_e,
                      int n_t,
                      float [:,:] bci_fifo,
                      float [:,:,:] templates,
                      int curr_idx,
                      int window,
                      unsigned int [:] matched_temp_idx_list,
                      unsigned int [:] peaked_elec_idx_list,
                      float [:] similarity,
                      int num_threads,
                      ) noexcept nogil:
    cdef int temp_idx
    cdef int temp
    cdef int elec
    cdef int elec_idx
    cdef int conn_elec
    cdef int t
    cdef float sim_temp

    # for elec in range(n_e): 
    for elec in prange(n_e, nogil=True, num_threads=num_threads):
        if peaked_elec_idx_list[elec]:
            for temp_idx in range(max_conn_temp):
                temp = elec_to_temp[elec][temp_idx]
                if temp == n_tm: # because elec_to_temp is padded with n_tm
                    break
                sim_temp = 0
                # check similarity for connected electrodes
                for elec_idx in range(max_conn_elec):
                    conn_elec = temp_to_elec[temp][elec_idx]
                    if conn_elec == n_e:
                        break
                    for t in range(n_t):
                        sim_temp = sim_temp + bci_fifo[conn_elec][(curr_idx-n_t+t+1)%window] * templates[temp][conn_elec][t]
                        
                similarity[temp] = sim_temp
    return
    
cdef void match_check(int max_conn_temp,
                      unsigned int [:,:] elec_to_temp,
                      int n_tm,
                      int n_e,
                      float [:] min_sps,
                      float [:] max_sps,
                      unsigned int [:] matched_temp_idx_list,
                      unsigned int [:] peaked_elec_idx_list,
                      float [:] best_amp_list,
                      float [:] similarity,
                      float n_scalar_neg,
                      int num_threads
                      ) noexcept nogil:
    cdef int temp_idx
    cdef int temp
    cdef int elec
    cdef float min_sps_temp
    cdef float max_sps_temp
    
    for elec in prange(n_e, nogil=True, num_threads=num_threads):
        if peaked_elec_idx_list[elec]:
            for temp_idx in range(max_conn_temp):
                temp = elec_to_temp[elec][temp_idx]
                if temp == n_tm:
                    break
                min_sps_temp = min_sps[temp]
                max_sps_temp = max_sps[temp]
                if similarity[temp] > min_sps_temp:
                    if similarity[temp] < max_sps_temp:
                        matched_temp_idx_list[temp] = 1
                        best_amp_list[temp] = similarity[temp] * n_scalar_neg
                        # FOR DEBUGGING
                        # best_amp_n[temp] = best_amp_list[temp] / norm_templates[temp]
            peaked_elec_idx_list[elec] = 0
    return

cdef void reset_history(int n_e,
                        float [:,:] bci_fifo,
                        int curr_idx,
                        int window,
                        int num_threads,
                        ) noexcept nogil:
    cdef int elec

    for elec in prange(n_e, nogil=True, num_threads=num_threads):
        bci_fifo[elec][(curr_idx+1) % window] = 0

cdef void update_history(int num_data_max,
                         int curr_idx,
                         int window,
                         unsigned int [:] elec_list,
                         float [:] data_list,
                         float [:,:] bci_fifo,
                         int n_e):
    cdef int i
    cdef int idx
    cdef int elec
    cdef float elec_data

    for i in range(num_data_max):
        idx = curr_idx % window
        elec = elec_list[i]
        elec_data = data_list[i]
        if elec == n_e:
            break
        bci_fifo[elec][idx] = elec_data

cdef void detect_peak(int n_e,
                      int num_threads,
                      float [:,:] bci_fifo,
                      int curr_idx,
                      int template_shift,
                      int window,
                      float [:] thresholds,
                      unsigned int [:] peaked_elec_idx_list,
                    ) noexcept nogil:
    cdef int elec
    cdef float curr_dat
    cdef float prev_dat_scalar
    cdef float next_dat_scalar
    cdef float elec_data

    for elec in prange(n_e, nogil=True,num_threads=num_threads):
        curr_dat = bci_fifo[elec][(curr_idx-template_shift)%window]
        if curr_dat < -thresholds[elec]:
            prev_dat_scalar = bci_fifo[elec][(curr_idx-template_shift-1)%window]
            next_dat_scalar = bci_fifo[elec][(curr_idx-template_shift+1)%window]
            if curr_dat < prev_dat_scalar and curr_dat < next_dat_scalar:
                peaked_elec_idx_list[elec] = 1

def spike_sorting_cpu_inner(int curr_idx,
                            int duration,
                            int window,
                            unsigned int [:,:] elec_to_temp,
                            unsigned int [:,:] temp_to_elec,
                            int max_conn_elec, 
                            int max_conn_temp, 
                            unsigned int [:] elec_list,
                            float [:] data_list,
                            int num_data_max,
                            int template_shift, 
                            float [:,:,:] templates,
                            float [:] norm_templates,
                            unsigned int [:] peaked_elec_idx_list,
                            unsigned int [:] matched_temp_idx_list,
                            float [:] best_amp_list, 
                            # float [:] best_amp_n,
                            float n_scalar_neg,
                            int n_e,
                            int n_t,
                            int n_tm,
                            float [:] thresholds,
                            float [:] min_sps,
                            float [:] max_sps,
                            float [:,:] bci_fifo,
                            float [:] similarity,
                            int num_threads,
                            unsigned int [:] total_valid_number,
                            unsigned int [:] result_spiketimes,
                            float [:] result_amplitudes
                            ):

    # reset bci_fifo for next timestep
    # check_event
    reset_history(n_e, bci_fifo, curr_idx, window, num_threads)

    # save data into bci_fifo
    # event-driven
    update_history(num_data_max, curr_idx, window, elec_list, data_list, bci_fifo, n_e)
    
    if curr_idx < n_t-1:
        return

    # perform peak detection
    # check event
    detect_peak(n_e, num_threads, bci_fifo, curr_idx, template_shift, window, thresholds, peaked_elec_idx_list)
    
    # event-driven
    calc_similarity(max_conn_temp, max_conn_elec, elec_to_temp, temp_to_elec, n_tm, n_e, n_t, 
                    bci_fifo, templates, curr_idx, window, 
                    matched_temp_idx_list, peaked_elec_idx_list, similarity, num_threads)
    # check event 
    match_check(max_conn_temp, elec_to_temp, n_tm, n_e, min_sps, max_sps,
                matched_temp_idx_list, peaked_elec_idx_list, best_amp_list, similarity, n_scalar_neg, num_threads)

    # event-driven
    subtract_overlap(matched_temp_idx_list, n_tm, n_e, n_t, curr_idx, template_shift, max_conn_elec, bci_fifo, best_amp_list, templates, temp_to_elec, similarity, window, num_threads, total_valid_number, result_spiketimes, result_amplitudes)

    return
