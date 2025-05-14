import sys
import numpy as np
import cupy as cp

import time
import timeit
import h5py
import scipy
import gc
import os

import spikeinterface.extractors as se
import circus

import configparser
from cupyx.profiler import benchmark, profile
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

        print("DATA SHAPE", data.shape)
        print("DATA NONZERO", np.count_nonzero(data))
        
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

def get_nodes(result_path, validating=False, shank_with=None):
    prb_filename = result_path + 'probe.prb'
    prb_myfile = open(prb_filename, 'r')
    nodes = []

    probe = {}
    probetext = prb_myfile.read()
    exec(probetext, probe)

    radius = int(probe['radius'])

    radius_factor = 0.5
    radius = int(radius_factor * float(radius))

    for key in list(probe['channel_groups'].keys()):
        # i => channel id
        for i in probe['channel_groups'][key]['channels']:
            nodes += [i]

    return np.array(nodes, dtype=np.int32)

def get_overlaps(result_path, erase=False, normalize=True, maxoverlap=True, half=False):
    file_out_suff = result_path + 'recording/recording'
    temp_filename = file_out_suff + '.templates.hdf5'
    overlap_filename = file_out_suff + '.overlap.hdf5'
    temp_myfile = h5py.File(temp_filename, 'r', libver='earliest')

    parallel_hdf5 = h5py.get_config().mpi
    N_e, N_t, nb_templates = temp_myfile.get('temp_shape')[:].ravel().astype(np.int32)
    hdf5_compress = True
    blosc_compress = False
    N_total = N_e # assume all electrodes are recorded
    duration = 2 * N_t - 1

    if os.path.exists(overlap_filename) and not erase:
        return h5py.File(overlap_filename, 'r', libver='earliest')
    else:
        if os.path.exists(overlap_filename) and erase:
            os.remove(overlap_filename)

    if maxoverlap:
        templates, _, _, _ = load_templates(file_out_suff)

    cluster_filename = file_out_suff + '.clusters.hdf5'
    if os.path.exists(cluster_filename):
        best_elec = h5py.File(cluster_filename, 'r', libver='earliest').get('electrodes')[:].ravel().astype(np.int32)
    else:
        raise Exception("No clusters found! Check suffix or run clustering?")
    
    nodes = get_nodes(result_path)
    N, N_tm = templates.shape

    norm_templates = temp_myfile.get('norms')[:]

    if half:
        N_tm //= 2

    inv_nodes = np.zeros(N_total, dtype=np.int32)
    inv_nodes[nodes] = np.arange(len(nodes))

    all_delays = np.arange(1, N_t + 1)

    if half:
        upper_bounds = N_tm
    else:
        upper_bounds = N_tm // 2

    to_explore = list(range(N_e))

    overlaps = {}
    overlaps['x'] = [np.zeros(0, dtype=np.uint32)]
    overlaps['y'] = [np.zeros(0, dtype=np.uint32)]
    overlaps['data'] = [np.zeros(0, dtype=np.float32)]
    overlaps['steps'] = []
    rows = np.arange(N_e*N_t)
    _srows = {'left': {}, 'right': {}}

    for idelay in all_delays:
        _srows['left'][idelay] = np.where(rows % N_t < idelay)[0]
        _srows['right'][idelay] = np.where(rows % N_t >= (N_t - idelay))[0]

    for ielec in to_explore:
        # best_elec : the electrodes with valid clusters
        local_idx = np.where(best_elec == ielec)[0]
        len_local = len(local_idx)

        if not half:
            local_idx = np.concatenate((local_idx, local_idx + upper_bounds))

        if len_local > 0:
            to_consider = np.arange(upper_bounds)
            if not half:
                to_consider = np.concatenate((to_consider, to_consider + upper_bounds))

            loc_templates = templates[:, local_idx].tocsr()
            loc_templates2 = templates[:, to_consider].tocsr()

            for idelay in all_delays:
                # tmp_1 left and right are correlated
                tmp_1 = loc_templates[_srows['left'][idelay]].T.tocsr()
                tmp_2 = loc_templates2[_srows['right'][idelay]]
                # _srows ?

                data = tmp_1.dot(tmp_2)

                dx, dy = data.nonzero()
                ddx = np.take(local_idx, dx).astype(np.uint32)
                ddy = np.take(to_consider, dy).astype(np.uint32)
                ones = np.ones(len(dx), dtype=np.uint32)
                overlaps['x'].append(ddx*N_tm + ddy)
                overlaps['y'].append((idelay - 1)*ones)
                overlaps['data'].append(data.data)
                #if idelay < N_t:
                #    overlaps['x'].append(ddy*N_tm + ddx)
                #    overlaps['y'].append((duration - idelay)*ones)
                #    overlaps['data'].append(data.data)

    sys.stderr.flush()
    print("Overlaps computed, now gathering data")

    hfile = h5py.File(filename, 'w', libver='earliest')
    over_shape = np.array([N_tm**2, duration], dtype=np.int32)
    hfile.create_dataset('over_shape', data=over_shape)

    for key in ['x', 'y', 'data']:
        data = np.concatenate(overlaps.pop(key))

        # We sort by x indices for faster retrieval later
        if key == 'x':
            indices = np.argsort(data).astype(np.int32)
        data = data[indices]

        if hdf5_compress:
            hfile.create_dataset('over_%s' %key, data=data, compression='gzip')
        else:
            hfile.create_dataset('over_%s' %key, data=data)

    # We need to gather the sparse arrays.
    hfile.close()

    gc.collect()

    if maxoverlap:
        filename = file_out_suff + '.overlap.hdf5'
        sys.stderr.flush()
        print("Overlaps gathered, now computing overlaps/lags")

        assert not half, "Error"
        N_half = N_tm // 2
        
        if os.path.exists(filename):
            c_overlap = h5py.File(filename, 'r')
            over_shape = c_overlap.get('over_shape')[:]
            N_over = int(np.sqrt(over_shape[0]))
            S_over = over_shape[1]
            c_overs = {}
            nb_data = 0

            over_x = c_overlap.get('over_x')[:]
            over_y = c_overlap.get('over_y')[:]
            over_data = c_overlap.get('over_data')[:]
            nb_data = len(over_x)

            c_overlap.close()

            data = np.zeros((nb_data,),dtype=np.float32)
            indices_x = np.zeros((nb_data,),dtype=np.int32)
            indices_y = np.zeros((nb_data,),dtype=np.int32)
            indices_sub = np.zeros((nb_data,),dtype=np.int32)
            indices_sorted = np.zeros((nb_data,),dtype=np.int32)

            data[:] = over_data
            indices_x[:] = over_x
            indices_y[:] = over_y
            indices_sub[:] = np.mod(over_x, N_over)
            indices_sorted[:] = np.argsort(indices_sub).astype(np.int32)

            over_x = indices_x
            over_y = indices_y
            over_data = data
            sub_over = indices_sub
            over_sorted = indices_sorted
            over_shape = over_shape
        else:
            print("No overlaps found! Check suffix?")
            sys.exit(0)

        to_explore = np.arange(N_half)[:]

        maxlags = np.zeros((len(to_explore), N_half), dtype=np.int32)
        maxoverlaps = np.zeros((len(to_explore), N_half), dtype=np.float32)

        res = []
        res2 = []
        for i in to_explore:
            res += [i * N_tm, i * N_tm + N_half]
            res2 += [i, i+1]

        bounds = np.searchsorted(over_x, res, 'left')
        bounds_2 = np.searchsorted(sub_over[over_sorted], res2, 'left')

        duration = over_shape[1] // 2
        mask_duration = over_y < duration

        for count, i in enumerate(to_explore):

            xmin, xmax = bounds[2*count:2*(count+1)]

            local_x = over_x[xmin:xmax] - i * N_tm
            local_y = over_y[xmin:xmax]
            local_data = over_data[xmin:xmax]

            xmin, xmax = bounds_2[2*count:2*(count+1)]
            nslice = over_sorted[xmin:xmax][mask_duration[over_sorted[xmin:xmax]]]

            local_x = np.concatenate((local_x, over_x[nslice] // N_tm))
            local_y = np.concatenate((local_y, (over_shape[1] - 1) - over_y[nslice]))
            local_data = np.concatenate((local_data, over_data[nslice]))

            data = scipy.sparse.csr_matrix((local_data, (local_x, local_y)), shape=(N_tm, over_shape[1]), dtype=np.float32)
            maxoverlaps[count, :] = data.max(1).toarray().flatten()[:N_half]
            maxlags[count, :] = N_t - 1 - np.array(data.argmax(1)).flatten()[:N_half]
            gc.collect()

        # Now we need to sync everything across nodes.
        line = np.arange(N_half)

        indices = list(np.arange(N_half))
        indices = np.argsort(indices).astype(np.int32)

        maxlags = maxlags[indices, :]
        maxlags[line, line] = 0

        #maxlags = np.maximum(maxlags, maxlags.T)
        #mask = np.tril(np.ones((N_half, N_half)), -1) > 0
        #maxlags[mask] *= -1

        gc.collect()


        indices = list(np.arange(N_half))
        indices = np.argsort(indices).astype(np.int32)

        maxoverlaps = maxoverlaps[indices, :]
        maxoverlaps[line, line] = 0
        #maxoverlaps = np.maximum(maxoverlaps, maxoverlaps.T)

        gc.collect()

        myfile2 = h5py.File(file_out_suff + '.templates.hdf5', 'r+', libver='earliest')

        for key in ['maxoverlap', 'maxlag', 'version']:
            if key in list(myfile2.keys()):
                myfile2.pop(key)

        if not normalize:
            maxoverlaps /= norm_templates[: N_half]
            maxoverlaps /= norm_templates[: N_half][:, np.newaxis]

        version = circus.__version__
        if version.find('+') > -1:
            version = version.split('+')[0]

        if version.find('/') > -1:
            version = '1.0.0'

        myfile2.create_dataset('version', data=np.array(version.split('.'), dtype=np.int32))

        if hdf5_compress:
            myfile2.create_dataset('maxlag',  data=maxlags, compression='gzip')
            myfile2.create_dataset('maxoverlap', data=maxoverlaps, compression='gzip')
        else:
            myfile2.create_dataset('maxlag',  data=maxlags)
            myfile2.create_dataset('maxoverlap', data=maxoverlaps)
        myfile2.close()

    gc.collect()

    return h5py.File(filename, 'r')

def load_overlaps(result_path):
    file_out_suff = result_path + 'recording/recording'
    filename = file_out_suff + '.overlap.hdf5'
    if os.path.exists(filename):
        c_overlap = h5py.File(filename, 'r')
        
        over_shape = c_overlap.get('over_shape')[:]
        N_over = np.int64(np.sqrt(over_shape[0]))
        S_over = over_shape[1]
        c_overs = [0 for _ in range(N_over)]
        nb_data = 0

        over_x = c_overlap.get('over_x')[:]
        over_y = c_overlap.get('over_y')[:]
        over_data = c_overlap.get('over_data')[:]
        nb_data = len(over_x) * 2
        factor = 2 * int(max(nb_data, (N_over + 1) ** 2))

        c_overlap.close()

        data = np.zeros((nb_data,),dtype=np.float32)
        indices = np.zeros((factor,), dtype=np.int32)

        global_offset_data = 0
        global_offset_ptr = 0
        local_nb_data = 0
        local_nb_ptr = 0

        duration = over_shape[1] // 2

        res = []
        res2 = []
        for i in range(N_over):
            res += [i * N_over, (i + 1) * N_over]
            res2 += [i, i+1]

        bounds = np.searchsorted(over_x, res, 'left')
        sub_over = np.mod(over_x, N_over)
        mask_duration = (over_y < duration)
        over_sorted = np.argsort(sub_over).astype(np.int32)
        bounds_2 = np.searchsorted(sub_over[over_sorted], res2, 'left')

        for i in range(N_over):

            xmin, xmax = bounds[2*i:2*(i+1)]
            local_x = over_x[xmin:xmax] - i * N_over
            local_y = over_y[xmin:xmax]
            local_data = over_data[xmin:xmax]

            xmin, xmax = bounds_2[2*i:2*(i+1)]
            nslice = over_sorted[xmin:xmax][mask_duration[over_sorted[xmin:xmax]]]

            local_x = np.concatenate((local_x, over_x[nslice] // N_over))
            local_y = np.concatenate((local_y, (over_shape[1] - 1) - over_y[nslice]))
            local_data = np.concatenate((local_data, over_data[nslice]))

            sparse_mat = scipy.sparse.csr_matrix((local_data, (local_x, local_y)), shape=(N_over, over_shape[1]))
            local_nb_data = len(sparse_mat.data)
            local_nb_ptr = len(sparse_mat.indptr)

            boundary_data = global_offset_data + local_nb_data
            boundary_ptr = global_offset_ptr + factor // 2

            data[global_offset_data:boundary_data] = sparse_mat.data
            indices[global_offset_data:boundary_data] = sparse_mat.indices
            indices[boundary_ptr:boundary_ptr + local_nb_ptr] = sparse_mat.indptr

            c_overs[i] = scipy.sparse.csr_matrix((N_over, S_over), dtype=np.float32)
            c_overs[i].data = data[global_offset_data:boundary_data]
            c_overs[i].indices = indices[global_offset_data:boundary_data]
            c_overs[i].indptr = indices[boundary_ptr:boundary_ptr + local_nb_ptr]
            c_overs[i] = c_overs[i].todense()
            global_offset_data += local_nb_data
            global_offset_ptr += local_nb_ptr

        gc.collect()

        return c_overs
    else:
        print("No overlaps found! Check suffix?")
        sys.exit(0)

def purge(file, pattern):
    dir = os.path.dirname(os.path.abspath(file))
    for f in os.listdir(dir):
        if f.find(pattern) > -1:
            os.remove(os.path.join(dir, f))


def spike_sorting_gpu_realtime(spatial_whitening, templates, norm_templates, thresholds, min_sps, max_sps, sparse, nb_chunks, chunk_size):
    elapsed = 0
    n_tm, n_t, n_e = templates.shape
    template_shift = (n_t - 1) // 2

    n_scalar = n_e * n_t
    n_scalar_inv = 1 / n_scalar
    print(n_e, n_t, template_shift)

    templates = cp.asarray(templates)
    thresholds = cp.asarray(-thresholds)
    min_sps = cp.asarray(cp.squeeze(min_sps))
    max_sps = cp.asarray(cp.squeeze(max_sps))

    temp_window = cp.arange(-template_shift, template_shift + 1)
    size_window = n_e * (2 * template_shift + 1)

    if debug:
        norm_templates = cp.asarray(norm_templates)

    if debug:
        result_spiketimes = cp.empty(0, dtype=cp.uint32)
        result_amplitudes = cp.empty(0, dtype=cp.float32)
        result_templates = cp.empty(0, dtype=cp.uint32)

    if sparse:
        # dimension of sparse matrix should be <= 2
        sparse_templates = cp.sparse.csr_matrix(templates.reshape(n_tm, n_scalar))
        
        # local_chunk = cp.sparse.csr_matrix(local_chunk)
        # thresholds = cp.sparse.csr_matrix(thresholds)
    
    padding = cp.empty((n_t,n_e), dtype=cp.float32)
    for chunk_idx in range(nb_chunks):
        offset = chunk_idx*chunk_size
        data_file = np.load(result_path+'recording.npy', allow_pickle=True, mmap_mode='r')
        data_file = data_file[offset:offset+chunk_size,:]
        local_chunk = np.dot(data_file, spatial_whitening)

        len_chunk = len(local_chunk)
        local_chunk = cp.asarray(local_chunk)
        if chunk_idx != 0:
            local_chunk = cp.concatenate((padding, local_chunk), axis=0)
        tic=timeit.default_timer()
        for curr_idx in range(template_shift, len_chunk - template_shift):
            # perform peak detection
            curr_dat = local_chunk[curr_idx, :]
            prev_dat_scalar = local_chunk[curr_idx - 1, :]
            next_dat_scalar = local_chunk[curr_idx + 1, :]
            # curr_dat = local_chunk[curr_idx-1:curr_idx+2, :]

            # Check if any electrodes spiked
            spiked = ((curr_dat < thresholds) & (curr_dat < prev_dat_scalar) & (curr_dat < next_dat_scalar)).any()
            # if sparse:
            #     spiked = (((curr_dat < thresholds).multiply(curr_dat < prev_dat_scalar)).multiply(curr_dat < next_dat_scalar))
            # else:
            #     spiked = ((curr_dat < thresholds) & (curr_dat < prev_dat_scalar) & (curr_dat < next_dat_scalar)).any()

            if not spiked:
            # if not spiked.size:
                # curr_idx += 1
                continue

            electrode_waveforms = local_chunk[curr_idx-template_shift:curr_idx+template_shift+1, :]

            if sparse:
                # print(cp.count_nonzero(electrode_waveforms), electrode_waveforms.size) # electrode_waveforms are not sparse
                dot_products = sparse_templates.multiply(electrode_waveforms.flatten()) # csr

                # todense here
                # dot_products = dot_products.todense()

                # reshape returns coo matrix -> should use todense or tocsr
                # dot_products = sparse_templates.multiply(electrode_waveforms.reshape((1, n_scalar)).todense()) # csr
                
                similarities = dot_products.sum(axis=1).flatten() # dense
                # similarities = cp.sum(dot_products, axis=[1])
            else:
                dot_products = cp.multiply(electrode_waveforms, templates)
                similarities = cp.sum(dot_products, axis=[1,2])

            # Check validity and find the best match
            valid_indices = cp.where((similarities > min_sps) & (similarities < max_sps))[0]

            if not len(valid_indices):
                continue # skip this value

            # Sum along the axis to get best_amp_list
            if sparse:
                # valid_dot_products = dot_products[valid_indices]
                # best_amp_list = valid_dot_products.sum(axis=1) * n_scalar_inv
                best_amp_list = dot_products.sum(axis=1)[valid_indices] * n_scalar_inv
            else:
                valid_dot_products = dot_products[valid_indices]
                best_amp_list = cp.sum(valid_dot_products, axis=[1,2]) * n_scalar_inv
                # best_amp_list = cp.sum(dot_products[valid_indices], axis=[1,2]) * n_scalar_inv

            if debug:
                if sparse:
                    best_amp_n = best_amp_list[-1][0] / norm_templates[valid_indices][-1]
                else:
                    best_amp_n = best_amp_list[-1] / norm_templates[valid_indices][-1]

            # subtract overlap for ALL valid template_indices
            if sparse:
                # sparse_list_template_best = sparse_templates[valid_indices]
                # local_chunk[curr_idx - template_shift:curr_idx + template_shift + 1, :] -= sparse_list_template_best.multiply(best_amp_list).sum(axis=0).reshape(n_t, n_e)

                # use original template -> remove reshape
                list_template_best = templates[valid_indices]
                local_chunk[curr_idx - template_shift:curr_idx + template_shift + 1, :] -= cp.sum(best_amp_list[:, cp.newaxis] * list_template_best, axis=0)
            else:
                list_template_best = templates[valid_indices]
                local_chunk[curr_idx - template_shift:curr_idx + template_shift + 1, :] -= cp.sum(best_amp_list[:, cp.newaxis, cp.newaxis] * list_template_best, axis=0)
                # local_chunk[start_idx:end_idx, :] -= cp.sum(cp.multiply(best_amp_list[:, cp.newaxis, cp.newaxis], list_template_best), axis=0)
            
            if curr_idx == len_chunk - template_shift - 1:
                padding = local_chunk[curr_idx - template_shift:curr_idx+template_shift+1,:]

            # Add matching to the result.
        
            if debug:
                result_spiketimes = cp.concatenate((result_spiketimes, cp.asarray([curr_idx] * len(valid_indices))))
                result_amplitudes = cp.concatenate((result_amplitudes, cp.asarray([best_amp_n] * len(valid_indices))))
                result_templates = cp.concatenate((result_templates, cp.asarray(valid_indices)))
            cp.cuda.Device().synchronize()
        toc=timeit.default_timer()
        elapsed += (toc-tic)
        del data_file

    if debug:
        print(result_spiketimes[:100])
        print(result_amplitudes[:100])
        print(result_templates[:100])

    sys.stderr.flush()
    return elapsed


if __name__ == '__main__':
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

    parser = argparse.ArgumentParser()
    parser.add_argument('--sparsify', type=float, dest='sparsify', default=-1, help='sparsify')
    args = parser.parse_args()

    sim_sparsify = args.sparsify

    if sim_sparsify < 0:
        sim_sparsify = properties.getfloat('training','sparsify')


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
    sim_num_gpu = properties.getint('general','num_gpu')
    sparse = properties.getboolean('general','gpu_sparse')
    chunk_size = properties.getint('general','chunk_size')

    result_path = result_path + data_name + "_" + str(sim_sparsify) + "/sorter_output/"

    # extract file extension from string 'dataset'
    filename, file_extension = os.path.splitext(dataset)
    filename = filename.split('/')[-1]
    print("filename :", filename)

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
    temp_path = result_path + 'recording/recording.templates.hdf5'

    # check user-provided simulation params
    print("-------------------------------")
    print("Simulation params")
    print(f"realtime : {sim_real_time}")
    print(f"spatial sparse : {sim_spatial_sparse}")
    print(f"valid once : {sim_valid_once}")
    print(f"spatial whitening : {sim_spatial_whitening}")
    print(f"sparsify : {sim_sparsify}")
    print(f"gpus : {sim_num_gpu}")
    print(f"gpu sparse : {sparse}")
    print(f"sign_peaks : {sim_sign_peaks}")
    print("-------------------------------")

    freq = recording.get_sampling_frequency()
    DT = 1000 / freq
    DURATION = recording.get_total_duration() # s
    duration = min(DURATION * sampling_rate, sim_duration)

    # DEBUGGING
    print("-------------------------------")
    print("Check recording parameters")
    print("frequency: ", freq)
    print("duration: ", duration)
    print("-------------------------------")
    nb_chunks = int((duration-1)//chunk_size + 1)


    sim_with_output = True
    file_out_suff = result_path + 'recording/recording'
    data_file = np.load(result_path+'recording.npy', allow_pickle=True, mmap_mode='r')

    templates, norm_templates, amp_limits, maxoverlap = load_templates(result_path)
    thresholds, spatial_whitening = get_accurate_thresholds(result_path)

    # Load required files

    N_tm, x = templates.shape
    c_overlap = get_overlaps(result_path)
    over_shape = c_overlap.get('over_shape')[:]
    n_over = int(np.sqrt(over_shape[0]))
    s_over = over_shape[1]
    s_center = s_over // 2
    n_tm = N_tm // 2

    # # If the number of overlaps is different from templates, we need to recompute them.
    if n_over != N_tm:
        print('Templates have been modified, recomputing the overlaps...')
        c_overlap = get_overlaps(result_path, erase=True)
        over_shape = c_overlap.get('over_shape')[:]
        n_over = int(np.sqrt(over_shape[0]))
        s_over = over_shape[1]
    
    # c_overs = load_overlaps(result_path)
    # nodes = get_nodes(result_path)

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

    templates = templates[:n_tm]
    norm_templates = norm_templates[:n_tm]
    sub_norm_templates = n_scalar * norm_templates

    min_scalar_products = amp_limits[:,0][:, np.newaxis]
    max_scalar_products = amp_limits[:,1][:, np.newaxis]
    min_sps = min_scalar_products * sub_norm_templates[:, np.newaxis]
    max_sps = max_scalar_products * sub_norm_templates[:, np.newaxis]

    spatial_sparse = True
    templates_spatial = np.zeros((n_tm, n_e, n_t), dtype=cp.float32)
    if spatial_sparse:
        elec_templates = templates.reshape(n_tm, n_e, -1)
        all_zeros = np.all(elec_templates == 0, axis=(1, 2))
        templates_spatial[~all_zeros] = elec_templates[~all_zeros]
    templates = templates_spatial

    del data_file
    # data_file = data_file[:int(duration),:]

    # spatial whitening
    # local_chunk = np.dot(data_file, spatial_whitening)

    # We do perform actual performance analysis here
    print("START BENCHMARK")
    time_sum = 0
    num_iter = 10
    with profile():
        for i in range(num_iter):
            elapsed = spike_sorting_gpu_realtime(spatial_whitening, templates.transpose(0, 2, 1), norm_templates, thresholds, min_sps, max_sps, sparse, nb_chunks, chunk_size)
            time_sum += elapsed
            print(f"Elapsed: {elapsed} (s)")    
    print("END BENCHMARK")
    print("Iter: ", num_iter)
    print(f"Average per iter: {time_sum/num_iter} (s)")

