import sys
import numpy as np

import time
import timeit
import h5py
import scipy

import os
import cProfile
import yappi

import spikeinterface.extractors as se

import circus

import configparser
import sorting
import argparse

np.random.seed(0)
debug = False


def load_templates(result_path):
    file_out_suff = result_path + 'recording/recording'
    file_name = file_out_suff+'.templates.hdf5'
    if os.path.exists(file_name):
        myfile = h5py.File(file_name, 'r', libver='earliest')
        temp_x = myfile.get('temp_x')[:].ravel()
        temp_y = myfile.get('temp_y')[:].ravel()
        temp_data = myfile.get('temp_data')[:].ravel()
        
        N_e, N_t, nb_templates = myfile.get('temp_shape')[:].ravel().astype(np.int32)

        sparse_mat = np.zeros((N_e*N_t, nb_templates), dtype=np.float32)
        sparse_mat[temp_x, temp_y] = temp_data

        norm_templates = myfile.get('norms')[:]

        for idx in range(sparse_mat.shape[1]):
            sparse_mat[:, idx] /= norm_templates[idx]

        sparse_mat = sparse_mat.T

        nb_data = sparse_mat.size
        data = np.zeros((nb_data,),dtype=np.float32)
        data = sparse_mat.ravel()
        
        # construct sparse templates
        templates = data.reshape(nb_templates, N_e*N_t)

        amp_limits = myfile.get('limits')[:]
        maxoverlap = myfile.get('maxoverlap')[:]
        
        return templates, norm_templates, amp_limits, maxoverlap
    else:
        print("No templates found! Check suffix?")
        sys.exit(0)

def get_accurate_thresholds(result_path, spike_thresh_min=0.9):
    file_out_suff = result_path + 'recording/recording'
    basis_filename = file_out_suff + '.basis.hdf5'
    temp_filename = file_out_suff + '.templates.hdf5'
    spike_thresh = 6
    basis_myfile = h5py.File(basis_filename, 'r', libver='earliest')
    temp_myfile = h5py.File(temp_filename, 'r', libver='earliest')
    thresholds = basis_myfile.get('thresholds')[:] * spike_thresh

    mads = basis_myfile.get('thresholds')[:]
    temp_x = temp_myfile.get('temp_x')[:].ravel()
    temp_y = temp_myfile.get('temp_y')[:].ravel()
    temp_data = temp_myfile.get('temp_data')[:].ravel()
    N_e, N_t, nb_templates = temp_myfile.get('temp_shape')[:].ravel().astype(np.int32)
    temp_myfile.close()
    templates = scipy.sparse.csc_matrix((temp_data, (temp_x, temp_y)), shape=(N_e * N_t, nb_templates))
    
    nb_temp = templates.shape[1] // 2
    spike_thresh = spike_thresh * spike_thresh_min

    for idx in range(nb_temp):
        template = templates[:, idx].toarray().ravel().reshape(N_e, N_t)
        a, b = np.unravel_index(template.argmin(), template.shape)
        value = -template[a, b]

        if thresholds[a] > value:
            thresholds[a] = max(spike_thresh * mads[a], value)
    
    # Get spatial_whitening
    spatial_whitening = np.ascontiguousarray(basis_myfile.get('spatial')[:])

    return thresholds, spatial_whitening

def purge(file, pattern):
    dir = os.path.dirname(os.path.abspath(file))
    for f in os.listdir(dir):
        if f.find(pattern) > -1:
            os.remove(os.path.join(dir, f))

def spike_sorting_cpu(elec_per_ts, data_per_ts, num_data_max,
                      elec_to_temp, temp_to_elec, max_conn_elec, max_conn_temp,
                      duration, window, n_scalar_neg, template_shift, templates,
                      peaked_elec_idx_list, matched_temp_idx_list, best_amp_list,
                      norm_templates, thresholds, min_sps, max_sps,
                      bci_fifo, similarity, num_threads,
                      result_spiketimes, result_amplitudes):
    print("Spike sorting CPU")
    n_tm, n_e, n_t = templates.shape

    print(n_e, n_tm, n_t, template_shift)

    local_borders = (template_shift, duration - template_shift)

    # FOR DEBUGGING
    total_valid_number = np.zeros((1,),dtype=np.uint32)
    
    for curr_idx in range(duration):
        if curr_idx >= local_borders[1]: # if out of chunk index
            break
        elec_list = np.array(elec_per_ts[curr_idx], dtype=np.uint32)
        data_list = np.array(data_per_ts[curr_idx], dtype=np.float32)

        # FOR DEBUGGING
        sorting.spike_sorting_cpu_inner(curr_idx, duration, window, elec_to_temp, temp_to_elec,
                                        max_conn_elec, max_conn_temp, elec_list, data_list, num_data_max,
                                        template_shift, templates, norm_templates,
                                        peaked_elec_idx_list, matched_temp_idx_list, best_amp_list,
                                        n_scalar_neg, n_e, n_t, n_tm, thresholds, min_sps, max_sps,
                                        bci_fifo, similarity, num_threads,
                                        total_valid_number, result_spiketimes, result_amplitudes)
    
    return



def main():
    # Load simulation parameters
    properties = configparser.ConfigParser()
    properties.read('ss.params')

    sim_mode = properties.get('general','mode')
    assert (sim_mode == "inference")

    # detection params
    sim_sign_peaks = properties.getint('detection', 'sign_peaks')
    sim_detect_threshold = properties.getint('detection', 'detect_threshold')

    # training params
    sim_spatial_whitening = properties.getboolean('training','spatial_whitening')
    sim_radius = properties.getfloat('training','radius') # NOTE : this parameter is also used in inference
    sim_radius_en = properties.getboolean('training','radius_en')

    # inference params
    sim_real_time = properties.getboolean('inference', 'real_time')
    sim_spatial_sparse = properties.getboolean('inference','spatial_sparse')
    sim_valid_once = properties.getboolean('inference','valid_once')
    sim_duration = properties.getint('inference','duration')

    # result params
    sim_snr_threshold = properties.getint('result', 'snr_threshold')

    dataset = properties.get('general','dataset')
    data_name = properties.get('general','data_name')
    result_path = properties.get('general','result_path')
    sampling_rate = properties.getint('general','sampling_rate')
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_thread', type=int, dest='num_thread', default=-1, help='num_thread')
    parser.add_argument('--sparsify', type=float, dest='sparsify', default=-1, help='sparsify')
    args = parser.parse_args()

    sim_num_workers      = args.num_thread
    sim_sparsify       = args.sparsify


    if sim_num_workers < 0:
        sim_num_workers = properties.getint('general','num_workers')
    if sim_sparsify < 0:
        sim_sparsify = properties.getfloat('training','sparsify')
    
    result_path = result_path + data_name + "_" + str(sim_sparsify) + "/sorter_output/"

    # extract file extension from string 'dataset'
    filename, file_extension = os.path.splitext(dataset)
    filename = filename.split('/')[-1]
    # print("filename :", filename)

    # declare
    recording = 0
    sorting_true = 0

    if file_extension == '.h5':
        print("MEArec dataset")
        # MEArec dataset - h5
        recording, sorting_true = se.read_mearec(dataset)
    elif file_extension == '.nwb':
        print("spikeinterface dataset")
        # Spikeinterface dataset - nwb
        sorting_true = se.NwbSortingExtractor(dataset)
        recording = se.NwbRecordingExtractor(dataset)
    else:
        print(f"not implemented for file extension: {file_extension}")
        print("this code does not support spikeforest dataset yet")
        exit()

    # check user-provided simulation params
    print("-------------------------------")
    print("Simulation params")
    print(f"realtime : {sim_real_time}")
    print(f"spatial sparse : {sim_spatial_sparse}")
    print(f"valid once : {sim_valid_once}")
    print(f"sparsify : {sim_sparsify}")
    print(f"spatial whitening : {sim_spatial_whitening}")
    print(f"cpus : {sim_num_workers}")
    print(f"sign_peaks : {sim_sign_peaks}")
    print("-------------------------------")

    freq = recording.get_sampling_frequency()
    DT = 1000 / freq
    DURATION = recording.get_total_duration() # s
    chan_ids = recording.get_channel_ids() # neuron_num
    chan_loc = recording.get_channel_locations()
    N_e = len(chan_ids)
    duration = min(DURATION * sampling_rate, sim_duration)

    # DEBUGGING
    print("-------------------------------")
    print("Check recording parameters")
    print("frequency: ", freq)
    print("duration: ", duration)
    # print("channel ids: ", chan_ids)
    print("-------------------------------")

    sim_with_output = True
    file_out_suff = result_path + 'recording/recording'
    data_file = np.load(result_path+'recording.npy', allow_pickle=True, mmap_mode='r')

    templates, norm_templates, amp_limits, maxoverlap = load_templates(result_path)
    thresholds, spatial_whitening = get_accurate_thresholds(result_path)

    N_tm, x = templates.shape
    n_tm = N_tm // 2

    if N_tm == 0:
        print("No templates present. Redo clustering?")
        sys.exit(0)

    n_t = 3 # Template width in ms
    n_t = int(sampling_rate * n_t * 1e-3)
    if np.mod(n_t, 2) == 0:
        n_t += 1

    purge(file_out_suff, '.data')

    # Preprocess templates
    n_e = data_file.shape[1]
    n_scalar = n_e * n_t
    n_scalar_neg = - (1/n_scalar)

    templates = templates[:n_tm]
    norm_templates = norm_templates[:n_tm]
    sub_norm_templates = n_scalar * norm_templates

    min_scalar_products = amp_limits[:,0][:, np.newaxis]
    max_scalar_products = amp_limits[:,1][:, np.newaxis]
    min_sps = np.array(min_scalar_products * sub_norm_templates[:, np.newaxis],dtype=np.float32)[:,0]
    max_sps = np.array(max_scalar_products * sub_norm_templates[:, np.newaxis],dtype=np.float32)[:,0]

    spatial_sparse = True
    templates_spatial = np.zeros((n_tm, n_e, n_t), dtype=np.float32)
    if spatial_sparse:
        elec_templates = templates.reshape(n_tm, n_e, -1)
        all_zeros = np.all(elec_templates == 0, axis=(1, 2))
        templates_spatial[~all_zeros] = elec_templates[~all_zeros]
    del elec_templates
    
    duration = int(duration)
    data_file = data_file[:duration,:]

    # spatial whitening
    local_chunk = np.dot(data_file, spatial_whitening)

    electrode_positions = np.load(result_path + 'electrode_positions.npy')
    assert(electrode_positions.shape[0] == n_e)

    adj_elec_dict = {}
    for peaked_elec in range(len(electrode_positions)):
        local_peaked_position = electrode_positions[peaked_elec]
        adj_elec_ids = []
        for elec in range(len(electrode_positions)):
            distance = np.linalg.norm(electrode_positions[elec] - local_peaked_position)
            if distance <= sim_radius:
                adj_elec_ids.append(elec)
        adj_elec_dict[peaked_elec] = adj_elec_ids

    template_shift = (n_t-1) // 2
    spike_boundary = (template_shift, duration - template_shift)
    
    if os.path.exists('elec_per_ts_'+str(N_e)+'_'+str(duration)+'_'+str(sim_sparsify)+'.npy') and os.path.exists('data_per_ts_'+str(N_e)+'_'+str(duration)+'_'+str(sim_sparsify)+'.npy'):
        elec_per_ts = np.load('elec_per_ts_'+str(N_e)+'_'+str(duration)+'_'+str(sim_sparsify)+'.npy',allow_pickle=True, mmap_mode='r')
        data_per_ts = np.load('data_per_ts_'+str(N_e)+'_'+str(duration)+'_'+str(sim_sparsify)+'.npy',allow_pickle=True, mmap_mode='r')
        num_data_max = elec_per_ts.shape[1]
        print("loaded data for streaming")
    else:
        if os.path.exists('external_spikes_'+str(N_e)+'_'+str(duration)+'_'+str(sim_sparsify)+'.npy'):
            print("loading existing external spikes data")
            external_spikes = np.load('external_spikes_'+str(N_e)+'_'+str(duration)+'_'+str(sim_sparsify)+'.npy', allow_pickle=True)
        else:
            print("generating external spikes data")
            # predefined valid timestep list
            valid_timestep_list = []
            external_spikes = []
            for timestep in range(duration):
                if timestep % 10000 == 0:
                    print(f"{timestep}/{duration}")
                if timestep < spike_boundary[0]:
                    continue
                if timestep >= spike_boundary[1]:
                    break
                to_send_neuron_ids = []
                for neuron_id in range(N_e):
                    peak_detected = False
                    curr_dat = - local_chunk[timestep,neuron_id]
                    if curr_dat > thresholds[neuron_id]:
                        prev_dat = - local_chunk[timestep - 1,neuron_id]
                        next_dat = - local_chunk[timestep + 1,neuron_id]
                        if curr_dat > prev_dat and curr_dat > next_dat:
                            peak_detected = True
                            # get list of adjacent neuron_ids(electrode ids)
                            adj_neuron_ids = adj_elec_dict[neuron_id]
                            # concat adj neuron ids
                            to_send_neuron_ids += adj_neuron_ids
                # union of adj_neuron_ids at this timestep
                to_send_neuron_ids = list(set(to_send_neuron_ids))

                for neuron_id in to_send_neuron_ids:
                    for sending_timestep in range(timestep - template_shift, timestep + template_shift + 1):
                        external_spikes.append((sending_timestep, neuron_id, local_chunk[sending_timestep][neuron_id]))

            print("start sorting external spikes")

            external_spikes = list(set(external_spikes))
            external_spikes.sort(key=lambda x: x[0])
            print("done sorting external spikes")
            np.save('external_spikes_'+str(N_e)+'_'+str(duration)+'_'+str(sim_sparsify)+'.npy', external_spikes)
            
        elec_per_ts = [[] for _ in range(duration)]
        data_per_ts = [[] for _ in range(duration)]
        for ext in external_spikes:
            ts, elec_id, elec_data = ext
            elec_per_ts[int(ts)].append(int(elec_id))
            data_per_ts[int(ts)].append(elec_data)
        n = len(max(elec_per_ts,key=len))
        elec_per_ts = np.array([x+[int(n_e)]*(n-len(x)) for x in elec_per_ts],dtype=np.uint32)
        num_data_max = n
        data_per_ts = np.array([x+[0]*(n-len(x)) for x in data_per_ts],dtype=np.float32)
        np.save('elec_per_ts_'+str(N_e)+'_'+str(duration)+'_'+str(sim_sparsify)+'.npy',elec_per_ts)
        np.save('data_per_ts_'+str(N_e)+'_'+str(duration)+'_'+str(sim_sparsify)+'.npy',data_per_ts)
        print("generated data for streaming")
        # external_spikes shape : (to_send_neuron_ids*n_t,3(ts,elec_id,elec_data))

    # generate connections
    temp_to_elec = [[] for _ in range(n_tm)]
    elec_to_temp = [[] for _ in range(n_e)]
    for tm in range(n_tm):
        for elec in range(n_e):
            if np.any(templates[tm][n_t*elec:n_t*(elec+1)]):
                temp_to_elec[tm].append(elec)
    n = len(max(temp_to_elec,key=len))
    temp_to_elec = np.array([x+[n_e]*(n-len(x)) for x in temp_to_elec],dtype=np.uint32)
    # print("TEMP TO ELEC", temp_to_elec.shape)
    max_conn_elec = n

    for elec in range(n_e):
        for tm in range(n_tm):
            if np.any(templates[tm][n_t*elec:n_t*(elec+1)]):
                elec_to_temp[elec].append(tm)
    n = len(max(elec_to_temp,key=len))
    elec_to_temp = np.array([x+[n_tm]*(n-len(x)) for x in elec_to_temp],dtype=np.uint32)
    # print("ELEC TO TEMP", elec_to_temp.shape)
    max_conn_temp = n


    templates = templates_spatial
    n_scalar = n_e * n_t

    window = 200
    bci_fifo = np.zeros((n_e, window), dtype=np.float32)
    #FOR DEBUGGING
    result_spiketimes = np.zeros((n_tm*100),dtype=np.uint32)
    result_amplitudes = np.zeros((n_tm*100),dtype=np.float32)

    peaked_elec_idx_list = np.zeros((n_e,),dtype=np.uint32)
    matched_temp_idx_list = np.zeros((n_tm,),dtype=np.uint32)
    best_amp_list = np.zeros((n_tm,),dtype=np.float32)
    best_amp_n = np.zeros((n_tm,),dtype=np.float32)
    similarity = np.zeros((n_tm),dtype=np.float32)
    num_threads = sim_num_workers

    time_sum = 0
    num_iter = 1
    yappi.set_clock_type("wall")
    yappi.start()
    for i in range(num_iter):
        print("START")
        tic=timeit.default_timer()
        spike_sorting_cpu(elec_per_ts, data_per_ts, num_data_max,
                  elec_to_temp, temp_to_elec, max_conn_elec, max_conn_temp, 
                  duration, window, n_scalar_neg, template_shift, templates,
                  peaked_elec_idx_list, matched_temp_idx_list, best_amp_list,
                  norm_templates, thresholds, min_sps, max_sps,
                  bci_fifo, similarity, num_threads,
                  result_spiketimes, result_amplitudes)
        toc=timeit.default_timer()
        print("END")
        print(f"Elapsed: {toc - tic} (s)")
        time_sum += (toc-tic)
    yappi.stop()
    
    threads = yappi.get_thread_stats()
    for thread in threads:
        print("Function stats for (%s) (%d)"%(thread.name, thread.id))
        yappi.get_func_stats(ctx_id=thread.id).print_all()

    print(f"Average per iter: {time_sum / num_iter} (s)")
    average = time_sum / num_iter / duration * 1000
    print(f"Average per ts: {average} (ms)")
    
    if debug:
        print(result_spiketimes[:100])

if __name__ == '__main__':
    main()
